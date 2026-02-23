#!/usr/bin/env python3
"""
LightRAG Benchmark: LoCoMo & SyllabusQA

Measures:
  - F1 score, Recall, Accuracy  (QA quality)
  - Insert / Query / Delete time (seconds)
  - Insert / Query / Delete token cost (via tiktoken cl100k_base)

Usage (small sample):
  python benchmark_locomo_syllabusqa.py \
      --dataset all \
      --sample-size 10 \
      --llm-model gpt-4o-mini \
      --embedding-model text-embedding-3-small \
      --api-key YOUR_KEY

Usage (full):
  python benchmark_locomo_syllabusqa.py \
      --dataset all \
      --sample-size 0 \
      --llm-model gpt-4o-mini \
      --embedding-model text-embedding-3-small \
      --api-key YOUR_KEY
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import string
import sys
import time
import unicodedata
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
                    raise RuntimeError(f"Multimodal embed HTTP {resp.status}: {body[:300]}")
                data = await resp.json()
                return data["data"]["embedding"]
        except (aiohttp.ClientError, RuntimeError) as exc:
            if attempt == 2:
                raise
            logger.warning(f"Multimodal embed retry {attempt+1}: {exc}")
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
            _multimodal_embed_single(session, t, model, url, headers)
            for t in texts
        ]
        embeddings = await asyncio.gather(*tasks)
    return np.array(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# F1 / Recall / Accuracy metrics (from LoCoMo evaluation code)
# ---------------------------------------------------------------------------

try:
    from nltk.stem import PorterStemmer

    _ps = PorterStemmer()
except Exception:
    _ps = None


def _normalize_answer(s: str) -> str:
    s = str(s).replace(",", "")

    def remove_articles(text):
        import re

        return re.sub(r"\b(a|an|the|and)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _stem(word: str) -> str:
    if _ps is not None:
        return _ps.stem(word)
    return word.lower()


def token_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = [_stem(w) for w in _normalize_answer(prediction).split()]
    gt_tokens = [_stem(w) for w in _normalize_answer(ground_truth).split()]
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens) if pred_tokens else 0.0
    recall = num_same / len(gt_tokens) if gt_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def multi_answer_f1(prediction: str, ground_truth: str) -> float:
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]
    return float(
        np.mean(
            [
                max(token_f1_score(p, gt) for p in predictions)
                for gt in ground_truths
            ]
        )
    )


def compute_f1_for_locomo(prediction: str, answer: str, category: int) -> float:
    answer = str(answer)
    if category == 3:
        answer = answer.split(";")[0].strip()

    if category in (2, 3, 4):
        return token_f1_score(prediction, answer)
    elif category == 1:
        return multi_answer_f1(prediction, answer)
    elif category == 5:
        if "no information available" in prediction.lower() or "not mentioned" in prediction.lower():
            return 1.0
        return 0.0
    return token_f1_score(prediction, answer)


def compute_f1_for_syllabusqa(prediction: str, answer: str) -> float:
    return token_f1_score(prediction, answer)


def compute_recall(prediction: str, answer: str) -> float:
    pred_tokens = set(_stem(w) for w in _normalize_answer(prediction).split())
    gt_tokens = set(_stem(w) for w in _normalize_answer(answer).split())
    if not gt_tokens:
        return 1.0
    return len(pred_tokens & gt_tokens) / len(gt_tokens)


def compute_accuracy(prediction: str, answer: str) -> float:
    return 1.0 if _normalize_answer(prediction) == _normalize_answer(str(answer)) else 0.0


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_locomo(
    data_path: str = "/home/wiang/locomo/data/locomo10.json",
    sample_size: int = 0,
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
                answer = qa.get("adversarial_answer", qa.get("answer", ""))
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

    if sample_size > 0:
        results = results[:sample_size]
        for r in results:
            if sample_size < len(r["qa_pairs"]):
                r["qa_pairs"] = r["qa_pairs"][:sample_size]
    return results


def load_syllabusqa(
    data_path: str = "/home/wiang/SyllabusQA/data/dataset_split/test.json",
    syllabi_dir: str = "/home/wiang/SyllabusQA/syllabi/syllabi_redacted/text",
    sample_size: int = 0,
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

    if sample_size > 0:
        results = results[:sample_size]
        for r in results:
            if sample_size < len(r["qa_pairs"]):
                r["qa_pairs"] = r["qa_pairs"][:sample_size]
    return results


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
) -> tuple[dict, PhaseMetrics]:
    """Run all QA queries under a single query mode and return aggregated results + metrics."""
    GLOBAL_TRACKER.reset()
    t0 = time.perf_counter()

    all_qa: list[QAResult] = []
    for doc in docs:
        for qa in doc["qa_pairs"]:
            question = qa["question"]
            answer_gt = str(qa["answer"])
            category = qa.get("category", 1)

            try:
                pred = await rag.aquery(
                    question,
                    param=QueryParam(mode=mode),
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
            acc = compute_accuracy(pred, answer_gt)

            all_qa.append(
                QAResult(
                    question=question,
                    answer_gt=answer_gt,
                    answer_pred=pred,
                    f1=f1,
                    recall=rec,
                    accuracy=acc,
                    category=category,
                )
            )

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

    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=llm_func,
        llm_model_name=args.llm_model,
        embedding_func=embedding_func,
        llm_model_max_async=args.max_async,
        embedding_func_max_async=args.max_async,
        max_parallel_insert=2,
    )
    await rag.initialize_storages()

    # --- Phase 1: INSERT (once) ---
    GLOBAL_TRACKER.reset()
    t0 = time.perf_counter()

    for doc in docs:
        await rag.ainsert(doc["doc_text"], ids=[doc["doc_id"]])

    insert_time = time.perf_counter() - t0
    insert_tokens = GLOBAL_TRACKER.snapshot()
    insert_metrics = PhaseMetrics("insert", insert_time, insert_tokens)
    logger.info(f"INSERT done: {insert_time:.2f}s  tokens={GLOBAL_TRACKER}")

    # --- Phase 2: QUERY (each mode) ---
    query_results: dict[str, dict] = {}
    for mode in query_modes:
        logger.info(f"--- Query mode: {mode} ---")
        agg, metrics = await _run_query_mode(rag, docs, mode, dataset_name)
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
        f"prompt={tok.get('prompt_tokens',0):,}  "
        f"comp={tok.get('completion_tokens',0):,}  "
        f"total={tok.get('total_tokens',0):,}  "
        f"calls={tok.get('call_count',0)}"
    )


def print_summary(summary: dict):
    ds = summary["dataset"]
    ins = summary["insert"]
    dlt = summary["delete"]

    print(f"\n{'='*78}")
    print(f"  Dataset: {ds}  |  Docs: {summary['num_docs']}  |  QA/mode: {summary['num_qa_per_mode']}")
    print(f"  Model: {summary['llm_model']}  |  Embedding: {summary['embedding_model']}")
    print(f"{'='*78}")
    print(f"  Insert:  {ins['time_sec']:.1f}s  | {_fmt_tokens(ins['tokens'])}")
    print(f"  Delete:  {dlt['time_sec']:.1f}s  | {_fmt_tokens(dlt['tokens'])}")

    print(f"\n  {'Mode':<8} {'F1':>8} {'Recall':>8} {'Acc':>8} {'Time(s)':>9} {'Tokens':>10}")
    print(f"  {'-'*55}")
    for mode, res in summary["query_modes"].items():
        qa = res["qa_metrics"]
        print(
            f"  {mode:<8} {qa['f1_mean']:>8.4f} {qa['recall_mean']:>8.4f} "
            f"{qa['accuracy_mean']:>8.4f} {res['time_sec']:>9.1f} "
            f"{res['tokens'].get('total_tokens',0):>10,}"
        )

    print(f"\n  Per-mode category breakdown:")
    for mode, res in summary["query_modes"].items():
        print(f"    [{mode}]")
        for cat, m in res.get("per_category", {}).items():
            print(
                f"      {cat}: n={m['count']}  F1={m['f1_mean']:.4f}  "
                f"Recall={m['recall_mean']:.4f}  Acc={m['accuracy_mean']:.4f}"
            )
    print(f"{'='*78}\n")


def parse_args():
    p = argparse.ArgumentParser(description="LightRAG Benchmark: LoCoMo & SyllabusQA")

    p.add_argument(
        "--dataset",
        choices=["locomo", "syllabusqa", "all"],
        default="all",
        help="Which dataset to benchmark",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of docs (LoCoMo) or syllabi (SyllabusQA) to use. "
        "Also limits QA pairs per doc when < total. 0 = full dataset.",
    )
    p.add_argument(
        "--query-modes",
        nargs="+",
        default=["local", "global", "hybrid", "naive", "mix"],
        choices=["local", "global", "hybrid", "naive", "mix"],
        help="Query modes to test (default: all five). Example: --query-modes local hybrid mix",
    )
    p.add_argument("--max-async", type=int, default=4)

    p.add_argument("--llm-model", required=True, help="LLM model name")
    p.add_argument("--llm-base-url", default=None, help="LLM API base URL")
    p.add_argument("--embedding-model", required=True, help="Embedding model name")
    p.add_argument("--embedding-base-url", default=None, help="Embedding API base URL")
    p.add_argument("--api-key", default=None, help="API key (shared for LLM + embedding if not set separately)")
    p.add_argument(
        "--multimodal-embedding",
        action="store_true",
        help="Use volcengine multimodal embedding API (/embeddings/multimodal) "
        "instead of standard OpenAI /embeddings. Required for doubao-embedding-vision.",
    )

    p.add_argument("--locomo-path", default="/home/wiang/locomo/data/locomo10.json")
    p.add_argument("--syllabusqa-path", default="/home/wiang/SyllabusQA/data/dataset_split/test.json")
    p.add_argument("--syllabi-dir", default="/home/wiang/SyllabusQA/syllabi/syllabi_redacted/text")

    p.add_argument("--output-dir", default="./benchmark_output", help="Directory for working data and results")
    p.add_argument("--output-json", default=None, help="Path to save JSON results (default: <output-dir>/results.json)")

    return p.parse_args()


async def async_main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    all_summaries = []

    if args.dataset in ("locomo", "all"):
        docs = load_locomo(args.locomo_path, args.sample_size)
        total_qa = sum(len(d["qa_pairs"]) for d in docs)
        logger.info(f"LoCoMo: {len(docs)} docs, {total_qa} QA pairs")
        summary = await run_benchmark_for_dataset("locomo", docs, args)
        print_summary(summary)
        all_summaries.append(summary)

    if args.dataset in ("syllabusqa", "all"):
        docs = load_syllabusqa(args.syllabusqa_path, args.syllabi_dir, args.sample_size)
        total_qa = sum(len(d["qa_pairs"]) for d in docs)
        logger.info(f"SyllabusQA: {len(docs)} docs, {total_qa} QA pairs")
        summary = await run_benchmark_for_dataset("syllabusqa", docs, args)
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
