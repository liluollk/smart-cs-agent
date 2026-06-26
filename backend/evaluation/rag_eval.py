"""
RAG 评测 — 召回率、准确率、Top-K 调优、相似度阈值。
独立于 API 运行，直接测试 KnowledgeBackend 的检索质量。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from knowledge.backends import KnowledgeBackend, PgVectorBackend

# 评测 Query → 期望命中的文档 source
RAG_TEST_CASES = [
    {
        "query": "理财产品A的收益率",
        "expected_sources": ["product_wealth_a.md"],
        "category": "产品查询",
    },
    {
        "query": "怎么开户",
        "expected_sources": ["account_opening.md", "account_materials.md"],
        "category": "流程查询",
    },
    {
        "query": "退款政策",
        "expected_sources": ["refund_policy.md"],
        "category": "政策查询",
    },
    {
        "query": "手续费多少",
        "expected_sources": ["trading_fees.md"],
        "category": "费用查询",
    },
    {
        "query": "忘记密码怎么办",
        "expected_sources": ["faq_password.md"],
        "category": "FAQ查询",
    },
    {
        "query": "怎么换银行卡",
        "expected_sources": ["faq_bankcard.md"],
        "category": "FAQ查询",
    },
    {
        "query": "基金赎回要多久",
        "expected_sources": ["fund_redemption.md"],
        "category": "赎回查询",
    },
    {
        "query": "意外险怎么买",
        "expected_sources": ["product_insurance_accident.md"],
        "category": "产品查询",
    },
    {
        "query": "投诉怎么处理",
        "expected_sources": ["complaint_process.md"],
        "category": "投诉查询",
    },
    {
        "query": "收益什么时候到账",
        "expected_sources": ["faq_income.md"],
        "category": "FAQ查询",
    },
    {
        "query": "未成年人可以开户吗",
        "expected_sources": ["faq_minor.md"],
        "category": "FAQ查询",
    },
    {
        "query": "定投怎么设置",
        "expected_sources": ["faq_auto_invest.md"],
        "category": "FAQ查询",
    },
    {
        "query": "交易限额多少",
        "expected_sources": ["trading_limits.md"],
        "category": "规则查询",
    },
    {
        "query": "如何注销账户",
        "expected_sources": ["faq_cancel.md"],
        "category": "FAQ查询",
    },
    {
        "query": "风险提示是什么",
        "expected_sources": ["risk_disclaimer.md"],
        "category": "合规查询",
    },
    {
        "query": "隐私怎么保护",
        "expected_sources": ["privacy_policy.md"],
        "category": "合规查询",
    },
    {
        "query": "开户需要什么材料",
        "expected_sources": ["account_materials.md", "account_opening.md"],
        "category": "流程查询",
    },
    {
        "query": "退款到账时间",
        "expected_sources": ["refund_timeline.md"],
        "category": "退款查询",
    },
    {
        "query": "客服电话",
        "expected_sources": ["contact_us.md"],
        "category": "客服查询",
    },
    {
        "query": "保险",
        "expected_sources": ["product_insurance_accident.md"],
        "category": "产品查询",
    },
]


@dataclass
class RAGResult:
    query: str
    top_k: int
    hits: int
    total_expected: int
    recall: float
    precision_at_k: float
    mrr: float
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    top_scores: list[float]
    retrieved_sources: list[str]
    expected_sources: list[str]
    duration_ms: float


def evaluate_rag(
        backend: KnowledgeBackend,
        top_k_values=None,
) -> dict:
    if top_k_values is None:
        top_k_values = [1, 3, 5, 10]
    results: list[RAGResult] = []

    for tc in RAG_TEST_CASES:
        query = tc["query"]
        expected = set(tc["expected_sources"])

        for top_k in top_k_values:
            start = time.time()
            docs = backend.search(query, top_k=top_k)
            duration = (time.time() - start) * 1000

            retrieved_sources = [d["source"] for d in docs]
            retrieved_set = set(retrieved_sources)
            hits = len(retrieved_set & expected)
            recall = hits / len(expected) if expected else 0.0
            precision = hits / len(retrieved_sources) if retrieved_sources else 0.0

            mrr = 0.0
            for i, src in enumerate(retrieved_sources):
                if src in expected:
                    mrr = 1.0 / (i + 1)
                    break

            top_scores = [d["score"] for d in docs[:top_k]]

            results.append(RAGResult(
                query=query,
                top_k=top_k,
                hits=hits,
                total_expected=len(expected),
                recall=round(recall, 3),
                precision_at_k=round(precision, 3),
                mrr=round(mrr, 3),
                hit_at_1=any(src in expected for src in retrieved_sources[:1]),
                hit_at_3=any(src in expected for src in retrieved_sources[:3]),
                hit_at_5=any(src in expected for src in retrieved_sources[:5]),
                top_scores=top_scores,
                retrieved_sources=retrieved_sources[:top_k],
                expected_sources=list(expected),
                duration_ms=round(duration, 2),
            ))

    return _summarize(results, top_k_values)


def _summarize(results: list[RAGResult], top_k_values: list[int]) -> dict:
    by_k = {}
    for k in top_k_values:
        k_results = [r for r in results if r.top_k == k]
        if not k_results:
            continue
        n = len(k_results)
        by_k[f"top_{k}"] = {
            "cases": n,
            "avg_recall": round(sum(r.recall for r in k_results) / n, 3),
            "avg_precision": round(sum(r.precision_at_k for r in k_results) / n, 3),
            "avg_mrr": round(sum(r.mrr for r in k_results) / n, 3),
            "hit_at_1_rate": round(sum(1 for r in k_results if r.hit_at_1) / n, 3),
            "hit_at_3_rate": round(sum(1 for r in k_results if r.hit_at_3) / n, 3),
            "hit_at_5_rate": round(sum(1 for r in k_results if r.hit_at_5) / n, 3),
            "avg_duration_ms": round(sum(r.duration_ms for r in k_results) / n, 2),
        }

    score_analysis = _analyze_scores(results)

    failures = []
    for r in results:
        if r.top_k == 5 and r.recall < 1.0:
            failures.append({
                "query": r.query,
                "expected": r.expected_sources,
                "retrieved": r.retrieved_sources,
                "recall": r.recall,
            })

    return {
        "total_queries": len(RAG_TEST_CASES),
        "by_top_k": by_k,
        "score_analysis": score_analysis,
        "failures": failures,
        "recommendation": _recommend(by_k, score_analysis),
    }


def _analyze_scores(results: list[RAGResult]) -> dict:
    k5 = [r for r in results if r.top_k == 5]
    all_scores = []
    for r in k5:
        all_scores.extend(r.top_scores)

    if not all_scores:
        return {}

    sorted_scores = sorted(all_scores)
    n = len(sorted_scores)
    return {
        "min": round(sorted_scores[0], 3),
        "max": round(sorted_scores[-1], 3),
        "avg": round(sum(all_scores) / n, 3),
        "p50": round(sorted_scores[n // 2], 3),
        "p90": round(sorted_scores[int(n * 0.9)], 3),
        "samples": sorted_scores[:10],
    }


def _recommend(by_k: dict[str, Any], score_analysis: dict[str, Any]) -> dict[str, Any]:
    recommendations = {}

    if "top_3" in by_k and "top_5" in by_k:
        t3 = by_k["top_3"]["avg_recall"]
        t5 = by_k["top_5"]["avg_recall"]
        if t5 - t3 < 0.05:
            recommendations["top_k"] = {
                "value": 3,
                "reason": f"top_3 召回率 {t3} 与 top_5 召回率 {t5} 差异 < 5%，top_3 更高效",
            }
        else:
            recommendations["top_k"] = {
                "value": 5,
                "reason": f"top_5 召回率 {t5} 显著高于 top_3 召回率 {t3}，建议使用 top_5",
            }

    if score_analysis:
        p50: float = score_analysis.get("p50", 0)
        min_score: float = score_analysis.get("min", 0)
        recommendations["score_threshold"] = dict(value=round(max(min_score, p50 * 0.3), 3),
                                                  reason=f"建议阈值设为 p50*0.3={round(p50 * 0.3, 3)}，平衡召回率和精确率")

    return recommendations


def main():
    import os
    from memory.long_term import LongTermMemory

    print("=" * 60)
    print("RAG 评测 — PgVectorBackend（混合检索：dense + sparse）")
    print("=" * 60)

    ltm = LongTermMemory(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", "5432")),
        database=os.getenv("PG_DATABASE", "smartcs"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", ""),
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        dashscope_base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    backend = PgVectorBackend(ltm)
    backend.seed()
    print(f"测试 Query 数: {len(RAG_TEST_CASES)}")
    print()

    report = evaluate_rag(backend, top_k_values=[1, 3, 5, 10])

    print("--- 按 Top-K 汇总 ---")
    for k, v in report["by_top_k"].items():
        print(f"  {k}:")
        print(f"    召回率: {v['avg_recall']:.1%}")
        print(f"    精确率: {v['avg_precision']:.1%}")
        print(f"    MRR:    {v['avg_mrr']:.3f}")
        print(f"    Hit@1:  {v['hit_at_1_rate']:.1%}")
        print(f"    Hit@3:  {v['hit_at_3_rate']:.1%}")
        print(f"    Hit@5:  {v['hit_at_5_rate']:.1%}")
        print(f"    平均耗时: {v['avg_duration_ms']:.1f}ms")

    print()
    print("--- 相似度分值分析 (top_5) ---")
    sa = report["score_analysis"]
    for k, v in sa.items():
        if k != "samples":
            print(f"  {k}: {v}")

    print()
    print("--- 推荐 ---")
    for k, v in report["recommendation"].items():
        print(f"  {k}: {v['value']} ({v['reason']})")

    if report["failures"]:
        print()
        print(f"--- 未召回 case ({len(report['failures'])} 个) ---")
        for f in report["failures"]:
            print(f"  ❌ {f['query']}")
            print(f"     期望: {f['expected']}")
            print(f"     实际: {f['retrieved']}")

    print()
    print("=" * 60)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
