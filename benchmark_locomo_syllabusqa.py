#!/usr/bin/env python3
"""
LightRAG Benchmark: LoCoMo & SyllabusQA

Measures:
  - F1 score, Recall, Accuracy  (QA quality)
  - Insert / Query / Delete time (seconds)
  - Insert / Query / Delete token cost (via tiktoken cl100k_base)
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import hmac
import json
import os
import re
import shutil
import string
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger

# ---------------------------------------------------------------------------
# Tiktoken-based token tracker (cl100k_base)
# ---------------------------------------------------------------------------

ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class TiktokenTracker:
    """Count tokens with tiktoken cl100k_base for every LLM / embedding call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0

    def add_usage(self, token_counts: dict):
        self.prompt_tokens += token_counts.get("prompt_tokens", 0)
        self.completion_tokens += token_counts.get("completion_tokens", 0)
        self.total_tokens += token_counts.get("prompt_tokens", 0) + token_counts.get(
            "completion_tokens", 0
        )
        self.call_count += 1

    def get_usage(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
        }

    def snapshot(self) -> dict:
        return self.get_usage()

    def __str__(self):
        u = self.get_usage()
        return (
            f"calls={u['call_count']}  prompt_tok={u['prompt_tokens']}  "
            f"comp_tok={u['completion_tokens']}  total_tok={u['total_tokens']}"
        )


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


# ---------------------------------------------------------------------------
# Wrapping LLM / embedding so every call is tracked with tiktoken
# ---------------------------------------------------------------------------

GLOBAL_TRACKER = TiktokenTracker()


async def tracked_llm_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    *,
    _model_name: str,
    _base_url: str | None,
    _api_key: str | None,
    **kwargs,
):
    if history_messages is None:
        history_messages = []

    prompt_text = (system_prompt or "") + prompt
    for m in history_messages:
        prompt_text += m.get("content", "")
    prompt_tok = count_tokens(prompt_text)

    result = await openai_complete_if_cache(
        _model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        base_url=_base_url,
        api_key=_api_key,
        **kwargs,
    )

    if isinstance(result, str):
        comp_tok = count_tokens(result)
    else:
        comp_tok = 0

    GLOBAL_TRACKER.add_usage(
        {"prompt_tokens": prompt_tok, "completion_tokens": comp_tok}
    )
    return result


async def tracked_embedding(
    texts: list[str],
    *,
    _model_name: str,
    _base_url: str | None,
    _api_key: str | None,
    _multimodal: bool = False,
    **kwargs,
) -> np.ndarray:
    prompt_tok = sum(count_tokens(t) for t in texts)
    if _multimodal:
        result = await _multimodal_embed_batch(
            texts, model=_model_name, base_url=_base_url, api_key=_api_key
        )
    else:
        result = await openai_embed.func(
            texts,
            model=_model_name,
            base_url=_base_url,
            api_key=_api_key,
            **kwargs,
        )
    GLOBAL_TRACKER.add_usage({"prompt_tokens": prompt_tok, "completion_tokens": 0})
    return result


async def _multimodal_embed_single(
    session: aiohttp.ClientSession,
    text: str,
    model: str,
    url: str,
    headers: dict,
) -> list[float]:
    """Call volcengine multimodal embedding API for a single text."""
    payload = {
        "model": model,
        "input": [{"type": "text", "text": text}],
    }
    for attempt in range(3):
        try:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(
                        f"Multimodal embed HTTP {resp.status}: {body[:300]}"
                    )
                data = await resp.json()
                return data["data"]["embedding"]
        except (aiohttp.ClientError, RuntimeError) as exc:
            if attempt == 2:
                raise
            logger.warning(f"Multimodal embed retry {attempt + 1}: {exc}")
            await asyncio.sleep(1 * (attempt + 1))
    return []


async def _multimodal_embed_batch(
    texts: list[str],
    model: str,
    base_url: str | None,
    api_key: str | None,
) -> np.ndarray:
    """Embed a batch of texts via volcengine multimodal embedding API (one call per text)."""
    url = (base_url or "https://ark.cn-beijing.volces.com/api/v3").rstrip("/")
    if not url.endswith("/embeddings/multimodal"):
        url += "/embeddings/multimodal"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    async with aiohttp.ClientSession() as session:
        tasks = [
            _multimodal_embed_single(session, t, model, url, headers) for t in texts
        ]
        embeddings = await asyncio.gather(*tasks)
    return np.array(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# VikingDB rerank (Volcengine HMAC-SHA256 auth)
# ---------------------------------------------------------------------------
def _volcengine_hmac_sha256(key: bytes, content: str) -> bytes:
    return hmac.new(key, content.encode("utf-8"), hashlib.sha256).digest()


def _volcengine_hash_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _volcengine_utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _volcengine_sign_headers(
    ak: str,
    sk: str,
    host: str,
    path: str,
    body: str,
    region: str = "cn-beijing",
    service: str = "vikingdb",
) -> dict[str, str]:
    """Build Volcengine HMAC-SHA256 Authorization header."""
    now = _volcengine_utc_now()
    x_date = now.strftime("%Y%m%dT%H%M%SZ")
    short_x_date = x_date[:8]
    x_content_sha256 = _volcengine_hash_sha256(body)

    signed_headers_str = ";".join(
        ["content-type", "host", "x-content-sha256", "x-date"]
    )
    canonical_request = "\n".join(
        [
            "POST",
            path,
            "",
            "content-type:application/json",
            f"host:{host}",
            f"x-content-sha256:{x_content_sha256}",
            f"x-date:{x_date}",
            "",
            signed_headers_str,
            x_content_sha256,
        ]
    )
    hashed_canonical = _volcengine_hash_sha256(canonical_request)
    credential_scope = f"{short_x_date}/{region}/{service}/request"
    string_to_sign = f"HMAC-SHA256\n{x_date}\n{credential_scope}\n{hashed_canonical}"

    k_date = _volcengine_hmac_sha256(sk.encode("utf-8"), short_x_date)
    k_region = _volcengine_hmac_sha256(k_date, region)
    k_service = _volcengine_hmac_sha256(k_region, service)
    k_signing = _volcengine_hmac_sha256(k_service, "request")
    signature = _volcengine_hmac_sha256(k_signing, string_to_sign).hex()

    authorization = (
        f"HMAC-SHA256 Credential={ak}/{credential_scope}, "
        f"SignedHeaders={signed_headers_str}, Signature={signature}"
    )
    return {
        "Content-Type": "application/json",
        "Host": host,
        "X-Date": x_date,
        "X-Content-Sha256": x_content_sha256,
        "Authorization": authorization,
    }


async def vikingdb_rerank(
    query: str,
    documents: list[str],
    top_n: int | None = None,
    extra_body: dict | None = None,
    *,
    ak: str,
    sk: str,
    host: str = "api-vikingdb.vikingdb.cn-beijing.volces.com",
    model_name: str = "doubao-seed-rerank",
    model_version: str | None = None,
    threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """Call VikingDB rerank API with Volcengine HMAC-SHA256 auth.

    Returns LightRAG-compatible list: [{"index": int, "relevance_score": float}, ...]
    """
    path = "/api/vikingdb/rerank"
    payload: dict[str, Any] = {
        "model_name": model_name,
        "query": [{"text": query}],
        "data": [[{"text": doc}] for doc in documents],
    }
    if model_version:
        payload["model_version"] = model_version

    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    headers = _volcengine_sign_headers(ak, sk, host, path, body)
    url = f"https://{host}{path}"

    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for attempt in range(3):
            try:
                async with session.post(url, data=body, headers=headers) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        raise RuntimeError(
                            f"VikingDB rerank HTTP {resp.status}: {err[:300]}"
                        )
                    resp_json = await resp.json()
                    break
            except (aiohttp.ClientError, RuntimeError) as exc:
                if attempt == 2:
                    raise
                logger.warning(f"VikingDB rerank retry {attempt + 1}: {exc}")
                await asyncio.sleep(1 * (attempt + 1))
                body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                headers = _volcengine_sign_headers(ak, sk, host, path, body)

    # Response format: {"code": "Success", "result": {"data": [{"id": 0, "score": 0.98}, ...]}}
    if resp_json.get("code") != "Success":
        logger.warning(
            f"VikingDB rerank non-success: {resp_json.get('code')}: {resp_json.get('message', '')}"
        )
        return []

    items = resp_json.get("result", {}).get("data", [])

    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        idx = item.get("id", item.get("index", 0))
        score = item.get("score", item.get("relevance_score", 0.0))
        if score >= threshold:
            results.append({"index": idx, "relevance_score": score})

    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    if top_n is not None and len(results) > top_n:
        results = results[:top_n]
    return results


# ---------------------------------------------------------------------------
# F1 / Recall / Accuracy metrics
# ---------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    """标准化答案文本：去标点、转小写、去冠词"""
    s = str(s).replace(",", "")

    def remove_articles(text):
        return re.sub(r"\b(a|an|the|and)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def check_refusal(text: str) -> bool:
    _REFUSAL_KEYWORDS = [
        "not mentioned",
        "no information",
        "cannot be answered",
        "none",
        "unknown",
        "don't know",
    ]
    return any(r in text.lower() for r in _REFUSAL_KEYWORDS)


def calculate_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def multi_answer_f1(prediction: str, ground_truth: str) -> float:
    """For multiple ground-truth answers (comma-separated), compute F1 for each and take the max."""
    if check_refusal(prediction) and check_refusal(ground_truth):
        return 1.0
    ground_truths = [g.strip() for g in ground_truth.split(",")]
    return float(max(calculate_f1(prediction, gt) for gt in ground_truths))


def compute_f1_for_locomo(prediction: str, answer: str, category: int) -> float:
    answer = str(answer)
    if category == 3:
        answer = answer.split(";")[0].strip()
    if category in (2, 3, 4):
        return calculate_f1(prediction, answer)
    elif category == 1:
        return multi_answer_f1(prediction, answer)
    elif category == 5:
        if check_refusal(prediction):
            return 1.0
        return 0.0
    return calculate_f1(prediction, answer)


def compute_f1_for_syllabusqa(prediction: str, answer: str) -> float:
    return calculate_f1(prediction, answer)


def compute_recall(prediction: str, answer: str) -> float:
    if check_refusal(prediction) and check_refusal(answer):
        return 1.0
    pred_tokens = set(normalize_answer(prediction).split())
    gt_tokens = set(normalize_answer(answer).split())
    if not gt_tokens:
        return 1.0
    return len(pred_tokens & gt_tokens) / len(gt_tokens)


async def llm_grader(
    _model_name: str,
    _base_url: str | None,
    _api_key: str | None,
    question: str,
    gold_answer: str,
    response: str,
    dataset_name: str = "Locomo",
) -> float:
    # 1. 根据 dataset_name 路由选择 Prompt
    if "Locomo" in dataset_name.lower():
        system_prompt = """
        You are an expert grader that determines if answers to questions match a gold standard answer
        """
        ACCURACY_PROMPT = f"""
    Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
        (1) a question (posed by one user to another user),
        (2) a 'gold' (ground truth) answer,
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember what I got the last time I went to Hawaii?
    Gold answer: A shell necklace
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

    Now it's time for the real question:
    Question: {question}
    Gold answer: {gold_answer}
    Generated answer: {response}

    First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
    Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

    Respond with JSON only: {{"is_correct": "CORRECT" or "WRONG", "reasoning": "your explanation"}}
    """
    else:
        # 通用 Prompt 或其他数据集的 Prompt
        system_prompt = """
        You are an expert grader that determines if an AI-generated answer matches the gold standard (ground truth) answer for a given question.
        """
        ACCURACY_PROMPT = f"""
        Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given:
            (1) A question
            (2) A 'gold' (ground truth) answer
            (3) A generated answer

        Grading rules:
        - If the generated answer correctly encompasses the core semantic meaning or facts of the gold answer, grade it as CORRECT.
        - If the generated answer contradicts the gold answer or misses the key factual information, it is WRONG.

        Question: {question}
        Gold answer: {gold_answer}
        Generated answer: {response}

        First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
        Respond with JSON only: {{"is_correct": "CORRECT" or "WRONG", "reasoning": "your explanation"}}
        """
    content = await openai_complete_if_cache(
        _model_name,
        ACCURACY_PROMPT,
        system_prompt=system_prompt,
        base_url=_base_url,
        api_key=_api_key
    )

    try:
        result = json.loads(content)
        label = result.get("is_correct", result.get("label", "WRONG"))
        return 1.0 if label.strip().lower() == "correct" else 0.0
    except json.JSONDecodeError:
        # 容错：防止 LLM 没按格式输出 JSON
        return 1.0 if "CORRECT" in content.upper() else 0.0


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
def _filter_docs(
    results: list[dict],
    doc_ids: list[str] | None,
    max_qa: int,
) -> list[dict]:
    """Filter and truncate loaded documents.

    Args:
        results: Full list of loaded docs.
        doc_ids: Explicit doc IDs or 0-based indices to keep. None = all.
        max_qa: Max QA pairs per doc (0 = all).
    """
    if doc_ids:
        id_set = set(doc_ids)
        idx_set: set[int] = set()
        for v in doc_ids:
            try:
                idx_set.add(int(v))
            except ValueError:
                pass
        filtered = [
            r for i, r in enumerate(results) if r["doc_id"] in id_set or i in idx_set
        ]
        if not filtered:
            available = [r["doc_id"] for r in results]
            logger.warning(
                f"No docs matched --doc-ids {doc_ids}. Available: {available}"
            )
        results = filtered

    if max_qa > 0:
        for r in results:
            if len(r["qa_pairs"]) > max_qa:
                r["qa_pairs"] = r["qa_pairs"][:max_qa]
    return results


def load_locomo(
    data_path: str = "./datas/locomo/data/locomo10.json",
    doc_ids: list[str] | None = None,
    max_qa: int = 0,
) -> list[dict]:
    """Return list of dicts: {doc_id, doc_text, qa_pairs: [{question, answer, category}]}"""
    raw = json.loads(Path(data_path).read_text())
    results = []
    for item in raw:
        conv = item["conversation"]
        parts = []
        for i in range(1, 100):
            dt_key = f"session_{i}_date_time"
            key = f"session_{i}"
            if key not in conv:
                break
            if dt_key in conv:
                parts.append(f"[{conv[dt_key]}]")
            for turn in conv[key]:
                parts.append(f"{turn['speaker']}: {turn['text']}")
            parts.append("")
        doc_text = "\n".join(parts)

        qa_pairs = []
        for qa in item["qa"]:
            category = qa.get("category", 1)
            if category == 5:
                # answer = qa.get("adversarial_answer", qa.get("answer", ""))
                continue  # Skip category 5 (adversarial) for now since it's not clear how to evaluate it with LLM grading
            else:
                answer = qa.get("answer", "")
            qa_pairs.append(
                {
                    "question": qa["question"],
                    "answer": str(answer),
                    "category": category,
                }
            )
        results.append(
            {
                "doc_id": item["sample_id"],
                "doc_text": doc_text,
                "qa_pairs": qa_pairs,
            }
        )

    return _filter_docs(results, doc_ids, max_qa)


def load_syllabusqa(
    data_path: str = "./datas/SyllabusQA/data/dataset_split/test.json",
    syllabi_dir: str = "./datas/SyllabusQA/syllabi/syllabi_redacted/text",
    doc_ids: list[str] | None = None,
    max_qa: int = 0,
) -> list[dict]:
    """Return list of dicts: {doc_id, doc_text, qa_pairs: [{question, answer, category}]}"""
    raw = json.loads(Path(data_path).read_text())

    by_syllabus: dict[str, list[dict]] = {}
    for item in raw:
        name = item["syllabus_name"]
        by_syllabus.setdefault(name, []).append(item)

    results = []
    for syllabus_name, qa_items in by_syllabus.items():
        txt_path = Path(syllabi_dir) / f"{syllabus_name}.txt"
        if not txt_path.exists():
            logger.warning(f"Syllabus text not found: {txt_path}")
            continue
        doc_text = txt_path.read_text(errors="replace")

        qa_pairs = []
        for item in qa_items:
            qa_pairs.append(
                {
                    "question": item["question"],
                    "answer": item["answer"],
                    "category": item.get("question_type", "general"),
                }
            )

        results.append(
            {
                "doc_id": syllabus_name,
                "doc_text": doc_text,
                "qa_pairs": qa_pairs,
            }
        )

    return _filter_docs(results, doc_ids, max_qa)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
@dataclass
class PhaseMetrics:
    name: str
    elapsed_sec: float = 0.0
    token_usage: dict = field(default_factory=dict)


@dataclass
class QAResult:
    question: str = ""
    answer_gt: str = ""
    answer_pred: str = ""
    f1: float = 0.0
    recall: float = 0.0
    accuracy: float = 0.0
    category: Any = None


async def detect_embedding_dim(embed_func, tracker: TiktokenTracker) -> int:
    old = tracker.snapshot()
    vectors = await embed_func(["dimension probe"])
    dim = vectors.shape[1] if len(vectors.shape) > 1 else len(vectors[0])
    # Restore tracker so probe tokens don't count
    tracker.prompt_tokens = old["prompt_tokens"]
    tracker.completion_tokens = old["completion_tokens"]
    tracker.total_tokens = old["total_tokens"]
    tracker.call_count = old["call_count"]
    return dim


def _aggregate_qa(all_qa: list[QAResult], dataset_name: str) -> dict:
    """Compute aggregate QA metrics from a list of QAResult."""
    f1_scores = [q.f1 for q in all_qa]
    recall_scores = [q.recall for q in all_qa]
    acc_scores = [q.accuracy for q in all_qa]

    categories = sorted(set(str(q.category) for q in all_qa))
    per_category = {}
    for cat in categories:
        cat_qa = [q for q in all_qa if str(q.category) == cat]
        per_category[cat] = {
            "count": len(cat_qa),
            "f1_mean": float(np.mean([q.f1 for q in cat_qa])),
            "recall_mean": float(np.mean([q.recall for q in cat_qa])),
            "accuracy_mean": float(np.mean([q.accuracy for q in cat_qa])),
        }

    return {
        "qa_metrics": {
            "f1_mean": float(np.mean(f1_scores)) if f1_scores else 0.0,
            "f1_std": float(np.std(f1_scores)) if f1_scores else 0.0,
            "recall_mean": float(np.mean(recall_scores)) if recall_scores else 0.0,
            "recall_std": float(np.std(recall_scores)) if recall_scores else 0.0,
            "accuracy_mean": float(np.mean(acc_scores)) if acc_scores else 0.0,
            "accuracy_std": float(np.std(acc_scores)) if acc_scores else 0.0,
        },
        "per_category": per_category,
        "qa_details": [
            {
                "question": q.question,
                "answer_gt": q.answer_gt,
                "answer_pred": q.answer_pred[:500],
                "f1": round(q.f1, 4),
                "recall": round(q.recall, 4),
                "accuracy": round(q.accuracy, 4),
                "category": q.category,
            }
            for q in all_qa
        ],
    }


async def _run_query_mode(
    rag: LightRAG,
    docs: list[dict],
    mode: str,
    dataset_name: str,
    _model_name: str,
    _base_url: str | None,
    _api_key: str | None,
    enable_rerank: bool = False,
    max_concurrent_queries: int = 4,
) -> tuple[dict, PhaseMetrics]:
    """Run all QA queries under a single query mode with concurrency control."""
    GLOBAL_TRACKER.reset()
    t0 = time.perf_counter()

    sem = asyncio.Semaphore(max_concurrent_queries)
    qa_inputs = [(qa, doc) for doc in docs for qa in doc["qa_pairs"]]

    async def _single_query(qa: dict, doc: dict) -> QAResult:
        question = qa["question"]
        answer_gt = str(qa["answer"])
        category = qa.get("category", 1)

        async with sem:
            try:
                pred = await rag.aquery(
                    question,
                    param=QueryParam(mode=mode, enable_rerank=enable_rerank),
                )
                if pred is None:
                    pred = ""
                pred = str(pred).strip()
            except Exception as e:
                logger.error(f"[{mode}] Query failed for '{question[:60]}': {e}")
                pred = ""

        if dataset_name == "locomo":
            f1 = compute_f1_for_locomo(pred, answer_gt, category)
        else:
            f1 = compute_f1_for_syllabusqa(pred, answer_gt)
        rec = compute_recall(pred, answer_gt)
        acc = await llm_grader(
            _model_name=_model_name,
            _base_url=_base_url,
            _api_key=_api_key,
            question=question,
            gold_answer=answer_gt,
            response=pred,
            dataset_name=dataset_name,
        )

        return QAResult(
            question=question,
            answer_gt=answer_gt,
            answer_pred=pred,
            f1=f1,
            recall=rec,
            accuracy=acc,
            category=category,
        )

    all_qa = await asyncio.gather(*[_single_query(qa, doc) for qa, doc in qa_inputs])

    elapsed = time.perf_counter() - t0
    tokens = GLOBAL_TRACKER.snapshot()
    metrics = PhaseMetrics(f"query_{mode}", elapsed, tokens)
    logger.info(f"QUERY [{mode}] done: {elapsed:.2f}s  tokens={GLOBAL_TRACKER}")

    agg = _aggregate_qa(all_qa, dataset_name)
    return agg, metrics


ALL_QUERY_MODES = ["local", "global", "hybrid", "naive", "mix"]


async def run_benchmark_for_dataset(
    dataset_name: str,
    docs: list[dict],
    args: argparse.Namespace,
) -> dict:
    """Run full insert → query(all modes) → delete benchmark for one dataset."""

    query_modes = args.query_modes
    logger.info(
        f"=== Benchmarking {dataset_name}: {len(docs)} documents, "
        f"modes={query_modes} ==="
    )

    working_dir = Path(args.output_dir) / f"rag_{dataset_name}"
    if working_dir.exists():
        shutil.rmtree(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    embed_func_partial = partial(
        tracked_embedding,
        _model_name=args.embedding_model,
        _base_url=args.embedding_base_url,
        _api_key=args.api_key,
        _multimodal=args.multimodal_embedding,
    )

    embedding_dim = await detect_embedding_dim(embed_func_partial, GLOBAL_TRACKER)
    logger.info(f"Detected embedding dim: {embedding_dim}")

    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        func=embed_func_partial,
    )

    llm_func = partial(
        tracked_llm_complete,
        _model_name=args.llm_model,
        _base_url=args.llm_base_url,
        _api_key=args.api_key,
    )

    rerank_func = None
    if args.enable_rerank and args.rerank_ak and args.rerank_sk:
        rerank_func = partial(
            vikingdb_rerank,
            ak=args.rerank_ak,
            sk=args.rerank_sk,
            host=args.rerank_host,
            model_name=args.rerank_model_name,
            model_version=args.rerank_model_version,
            threshold=args.rerank_threshold,
        )
        logger.info(
            f"Rerank enabled: {args.rerank_model_name} "
            f"(version={args.rerank_model_version}) on {args.rerank_host}"
        )

    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=llm_func,
        llm_model_name=args.llm_model,
        embedding_func=embedding_func,
        rerank_model_func=rerank_func,
        llm_model_max_async=args.max_async,
        embedding_func_max_async=args.max_async,
        max_parallel_insert=args.max_parallel_insert,
    )
    await rag.initialize_storages()

    # --- Phase 1: INSERT (batch) ---
    GLOBAL_TRACKER.reset()
    t0 = time.perf_counter()

    doc_texts = [doc["doc_text"] for doc in docs]
    doc_ids = [doc["doc_id"] for doc in docs]
    await rag.ainsert(doc_texts, ids=doc_ids)

    insert_time = time.perf_counter() - t0
    insert_tokens = GLOBAL_TRACKER.snapshot()
    insert_metrics = PhaseMetrics("insert", insert_time, insert_tokens)
    logger.info(f"INSERT done: {insert_time:.2f}s  tokens={GLOBAL_TRACKER}")

    # --- Phase 2: QUERY (each mode) ---
    query_results: dict[str, dict] = {}
    for mode in query_modes:
        logger.info(f"--- Query mode: {mode} ---")
        agg, metrics = await _run_query_mode(
            rag,
            docs,
            mode,
            dataset_name,
            _model_name=args.llm_model,
            _base_url=args.llm_base_url,
            _api_key=args.api_key,
            enable_rerank=args.enable_rerank,
            max_concurrent_queries=args.max_concurrent_queries,
        )
        query_results[mode] = {
            **agg,
            "time_sec": round(metrics.elapsed_sec, 3),
            "tokens": metrics.token_usage,
        }

    # --- Phase 3: DELETE ---
    GLOBAL_TRACKER.reset()
    t0 = time.perf_counter()

    for doc in docs:
        try:
            await rag.adelete_by_doc_id(doc["doc_id"])
        except Exception as e:
            logger.error(f"Delete failed for doc {doc['doc_id']}: {e}")

    delete_time = time.perf_counter() - t0
    delete_tokens = GLOBAL_TRACKER.snapshot()
    delete_metrics = PhaseMetrics("delete", delete_time, delete_tokens)
    logger.info(f"DELETE done: {delete_time:.2f}s  tokens={GLOBAL_TRACKER}")

    await rag.finalize_storages()

    summary = {
        "dataset": dataset_name,
        "num_docs": len(docs),
        "num_qa_per_mode": sum(len(d["qa_pairs"]) for d in docs),
        "llm_model": args.llm_model,
        "embedding_model": args.embedding_model,
        "insert": {
            "time_sec": round(insert_metrics.elapsed_sec, 3),
            "tokens": insert_metrics.token_usage,
        },
        "query_modes": query_results,
        "delete": {
            "time_sec": round(delete_metrics.elapsed_sec, 3),
            "tokens": delete_metrics.token_usage,
        },
    }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _fmt_tokens(tok: dict) -> str:
    return (
        f"prompt={tok.get('prompt_tokens', 0):,}  "
        f"comp={tok.get('completion_tokens', 0):,}  "
        f"total={tok.get('total_tokens', 0):,}  "
        f"calls={tok.get('call_count', 0)}"
    )


def print_summary(summary: dict):
    ds = summary["dataset"]
    ins = summary["insert"]
    dlt = summary["delete"]

    print(f"\n{'=' * 78}")
    print(
        f"  Dataset: {ds}  |  Docs: {summary['num_docs']}  |  QA/mode: {summary['num_qa_per_mode']}"
    )
    print(
        f"  Model: {summary['llm_model']}  |  Embedding: {summary['embedding_model']}"
    )
    print(f"{'=' * 78}")
    print(f"  Insert:  {ins['time_sec']:.1f}s  | {_fmt_tokens(ins['tokens'])}")
    print(f"  Delete:  {dlt['time_sec']:.1f}s  | {_fmt_tokens(dlt['tokens'])}")

    print(
        f"\n  {'Mode':<8} {'F1':>8} {'Recall':>8} {'Acc':>8} {'Time(s)':>9} {'Tokens':>10}"
    )
    print(f"  {'-' * 55}")
    for mode, res in summary["query_modes"].items():
        qa = res["qa_metrics"]
        print(
            f"  {mode:<8} {qa['f1_mean']:>8.4f} {qa['recall_mean']:>8.4f} "
            f"{qa['accuracy_mean']:>8.4f} {res['time_sec']:>9.1f} "
            f"{res['tokens'].get('total_tokens', 0):>10,}"
        )

    print(f"\n  Per-mode category breakdown:")
    for mode, res in summary["query_modes"].items():
        print(f"    [{mode}]")
        for cat, m in res.get("per_category", {}).items():
            print(
                f"      {cat}: n={m['count']}  F1={m['f1_mean']:.4f}  "
                f"Recall={m['recall_mean']:.4f}  Acc={m['accuracy_mean']:.4f}"
            )
    print(f"{'=' * 78}\n")


def parse_args():
    p = argparse.ArgumentParser(description="LightRAG Benchmark: LoCoMo & SyllabusQA")

    p.add_argument(
        "--dataset",
        nargs="+",
        choices=["locomo", "syllabusqa"],
        default=["locomo", "syllabusqa"],
        help="Datasets to benchmark (default: both). Example: --dataset locomo syllabusqa",
    )
    p.add_argument(
        "--doc-ids",
        nargs="+",
        default=None,
        help="Specific doc IDs or 0-based indices to test. "
        "LoCoMo IDs: conv-26 .. conv-50; SyllabusQA IDs: syllabus names. "
        "Example: --doc-ids conv-26 conv-42  or  --doc-ids 0 3",
    )
    p.add_argument(
        "--max-qa",
        type=int,
        default=0,
        help="Max QA pairs per doc (0 = all). Use to limit QA independently of --sample-size.",
    )
    p.add_argument(
        "--query-modes",
        nargs="+",
        default=["local", "global", "hybrid", "naive", "mix"],
        choices=["local", "global", "hybrid", "naive", "mix"],
        help="Query modes to test (default: all five). Example: --query-modes local hybrid mix",
    )
    p.add_argument(
        "--max-async",
        type=int,
        default=8,
        help="Max concurrent LLM/embedding calls inside LightRAG (default: 8)",
    )
    p.add_argument(
        "--max-parallel-insert",
        type=int,
        default=4,
        help="Max docs processed in parallel during insert (default: 4, max recommended: 10)",
    )
    p.add_argument(
        "--max-concurrent-queries",
        type=int,
        default=4,
        help="Max QA queries run concurrently (default: 4)",
    )

    p.add_argument("--llm-model", required=True, help="LLM model name")
    p.add_argument("--llm-base-url", default=None, help="LLM API base URL")
    p.add_argument("--embedding-model", required=True, help="Embedding model name")
    p.add_argument("--embedding-base-url", default=None, help="Embedding API base URL")
    p.add_argument(
        "--api-key",
        default=None,
        help="API key (shared for LLM + embedding if not set separately)",
    )
    p.add_argument(
        "--multimodal-embedding",
        action="store_true",
        help="Use volcengine multimodal embedding API (/embeddings/multimodal) "
        "instead of standard OpenAI /embeddings. Required for doubao-embedding-vision.",
    )
    p.add_argument(
        "--enable-rerank",
        action="store_true",
        default=False,
        help="Enable reranking during query. "
        "When set, also provide --rerank-ak and --rerank-sk for VikingDB rerank.",
    )
    p.add_argument(
        "--rerank-ak", default=None, help="Volcengine AK for VikingDB rerank"
    )
    p.add_argument(
        "--rerank-sk", default=None, help="Volcengine SK for VikingDB rerank"
    )
    p.add_argument(
        "--rerank-host",
        default="api-vikingdb.vikingdb.cn-beijing.volces.com",
        help="VikingDB rerank API host",
    )
    p.add_argument(
        "--rerank-model-name", default="doubao-seed-rerank", help="Rerank model name"
    )
    p.add_argument(
        "--rerank-model-version",
        default=None,
        help="Rerank model version (e.g. 251028)",
    )
    p.add_argument(
        "--rerank-threshold",
        type=float,
        default=0.1,
        help="Min rerank score threshold (default: 0.1)",
    )

    p.add_argument("--locomo-path", default="./datas/locomo/data/locomo10.json")
    p.add_argument(
        "--syllabusqa-path",
        default="./datas/SyllabusQA/data/dataset_split/test.json",
    )
    p.add_argument(
        "--syllabi-dir", default="./datas/SyllabusQA/syllabi/syllabi_redacted/text"
    )

    p.add_argument(
        "--output-dir",
        default="./benchmark_output",
        help="Directory for working data and results",
    )
    p.add_argument(
        "--output-json",
        default=None,
        help="Path to save JSON results (default: <output-dir>/results.json)",
    )

    return p.parse_args()


async def async_main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    if args.enable_rerank and not (args.rerank_ak and args.rerank_sk):
        logger.warning(
            "--enable-rerank is set but --rerank-ak / --rerank-sk are missing. "
            "Rerank will be disabled."
        )
        args.enable_rerank = False

    all_summaries = []

    for ds in args.dataset:
        if ds == "locomo":
            docs = load_locomo(
                args.locomo_path,
                doc_ids=args.doc_ids,
                max_qa=args.max_qa,
            )
            total_qa = sum(len(d["qa_pairs"]) for d in docs)
            logger.info(f"LoCoMo: {len(docs)} docs, {total_qa} QA pairs")
        elif ds == "syllabusqa":
            docs = load_syllabusqa(
                args.syllabusqa_path,
                args.syllabi_dir,
                doc_ids=args.doc_ids,
                max_qa=args.max_qa,
            )
            total_qa = sum(len(d["qa_pairs"]) for d in docs)
            logger.info(f"SyllabusQA: {len(docs)} docs, {total_qa} QA pairs")
        else:
            continue

        if docs:
            summary = await run_benchmark_for_dataset(ds, docs, args)
            print_summary(summary)
            all_summaries.append(summary)

    out_path = args.output_json or os.path.join(args.output_dir, "results.json")
    Path(out_path).write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False))
    logger.info(f"Results saved to {out_path}")

    return all_summaries


def main():
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
