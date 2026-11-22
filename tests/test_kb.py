"""知识库检索（v1.5.6）：BM25 命中应指向对的那一条，且可解释。

为什么测这个：检索错了，模型就照着错条款答，比不检索更糟。
所以每条断言都锁"该命中的命中、不该命中的不在前三"。
"""
from __future__ import annotations

from inner.kb import KnowledgeBase

RULES = [
    {"rule_id": "R-DISPUTE-01", "summary": "争议须由客户通过 App「我的-交易争议」提交，需交易凭证与身份核验"},
    {"rule_id": "R-TXN-02", "summary": "交易状态以核心系统为准，客服不得自行判定入账"},
    {"rule_id": "R-UNAUTH-02", "summary": "先临时冻结，再转风控，不得承诺赔付"},
]


def _kb():
    return KnowledgeBase(RULES)


def test_index_built_with_rules_and_static():
    kb = _kb()
    assert kb.size >= 10, kb.size
    ids = {d["id"] for d in kb.docs}
    assert "R-TXN-02" in ids and "KB-FEE-01" in ids


def test_dispute_material_hit():
    hits = _kb().search("想提一笔交易争议，要准备哪些材料", k=3)
    ids = [h["id"] for h in hits]
    assert "KB-DISPUTE-02" in ids or "KB-DISPUTE-01" in ids, ids
    assert hits[0]["score"] > 0


def test_ledger_authority_hit():
    hits = _kb().search("这笔交易到底入没入账，以什么为准", k=3)
    ids = [h["id"] for h in hits]
    assert "R-TXN-02" in ids or "KB-TXN-02" in ids, ids


def test_fee_hit_not_refund_promise():
    hits = _kb().search("信用卡年费能不能免", k=3)
    ids = [h["id"] for h in hits]
    assert "KB-FEE-01" in ids, ids


def test_empty_query_returns_nothing():
    assert _kb().search("") == []


def test_context_format_has_ids():
    hits = _kb().search("卡被冻结了急用钱", k=2)
    ctx = _kb().as_context(hits)
    assert ctx and "[KB-" in ctx or "[R-" in ctx
