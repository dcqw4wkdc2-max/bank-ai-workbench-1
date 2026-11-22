"""知识库检索（v1.5.6）：中文 2-gram BM25，零依赖、可解释。

为什么不用向量：
- 现有知识量小（规则 8 条 + 静态切片 16 条），全量与检索差别不大，向量库带来的
  召回错误反而更难追责；
- 演示环境没有 embedding 服务，引入即多一个外部依赖与费用；
- BM25 能打印"命中哪些词、命中哪一条、得分多少"，评委与审计都看得懂。

检索结果只用于**增强**提示词：命中条款连同编号一起给模型，要求回答引用编号；
命中为空或异常时退回"全量规则"路径，不影响已验证的链路。
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

_DATA = Path(__file__).with_name("kb_data.json")

# 中文按 2-gram 切分：短文本检索够用，且不需要分词词典
_CJK = re.compile(r"[\u4e00-\u9fff]")
_WORD = re.compile(r"[A-Za-z0-9_]{2,}")


def _tokens(text: str) -> list[str]:
    text = (text or "").lower()
    toks: list[str] = []
    buf: list[str] = []
    for ch in text:
        if _CJK.match(ch):
            buf.append(ch)
        else:
            if len(buf) >= 2:
                toks += ["".join(buf[i:i + 2]) for i in range(len(buf) - 1)]
            buf = []
            toks += _WORD.findall(ch)
    if len(buf) >= 2:
        toks += ["".join(buf[i:i + 2]) for i in range(len(buf) - 1)]
    elif buf:
        toks.append(buf[0])
    return toks


class KnowledgeBase:
    """轻量知识库：rules 表 + 静态切片，进内存建索引。"""

    def __init__(self, static_entries: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: list[dict] = []
        self._tf: list[Counter] = []
        self._len: list[int] = []
        self._df: Counter = Counter()
        self._load(static_entries)

    # ---------- 建索引 ----------
    def _load(self, static_entries: list[dict]) -> None:
        raw = list(static_entries)
        try:
            data = json.loads(_DATA.read_text(encoding="utf-8"))
            raw += data.get("entries", [])
        except Exception:
            pass  # 静态库缺失不影响规则检索
        for e in raw:
            rid = str(e.get("id") or e.get("rule_id") or "").strip()
            text = str(e.get("text") or e.get("summary") or "").strip()
            title = str(e.get("title") or "").strip()
            if not rid or not text:
                continue
            if any(d["id"] == rid for d in self.docs):
                continue  # 规则表优先，避免重复
            self.docs.append({"id": rid, "title": title, "text": text})
        for d in self.docs:
            toks = _tokens(d["title"] + " " + d["text"])
            tf = Counter(toks)
            self._tf.append(tf)
            self._len.append(len(toks))
            for t in tf:
                self._df[t] += 1
        self._avg = (sum(self._len) / len(self._len)) if self._len else 0.0

    @property
    def size(self) -> int:
        return len(self.docs)

    # ---------- 检索 ----------
    def search(self, query: str, k: int = 3) -> list[dict]:
        if not self.docs or not query:
            return []
        n = len(self.docs)
        q = Counter(_tokens(query))
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            dl = self._len[i] or 1
            for t, qf in q.items():
                f = tf.get(t, 0)
                if not f:
                    continue
                df = self._df.get(t, 0) or 1
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                score += qf * idf * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / (self._avg or 1)))
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        out = []
        for score, i in scored[:k]:
            d = self.docs[i]
            out.append({"id": d["id"], "title": d["title"], "text": d["text"],
                        "score": round(score, 3)})
        return out

    def as_context(self, hits: list[dict]) -> str:
        return "\n".join(f"- [{h['id']}] {h['text']}" for h in hits)


_CACHE: KnowledgeBase | None = None


def get_kb(rules: list[dict] | None = None) -> KnowledgeBase:
    """进程内单例；rules 变化（如规则发布）时由调用方传新列表即可重建。"""
    global _CACHE
    if _CACHE is None:
        _CACHE = KnowledgeBase(rules or [])
    return _CACHE


def rebuild(rules: list[dict]) -> KnowledgeBase:
    global _CACHE
    _CACHE = KnowledgeBase(rules)
    return _CACHE
