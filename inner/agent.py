"""微光「牛马仕」(Bull Hermes) 智能体适配层 + 模型连接器（两者刻意分离）。

牛马仕是微光工作室自研、唯一可私有化部署到客户内网的智能体产品——
正因为部署在客户内网，它才被允许接触未出域的上下文。
分层的目的：换模型不需要改 Agent，换 Agent 不需要改网络。
两者都有各自独立的 Mock，联调时分别验证——任务书明确禁止把两者合并成
一个"模型 Mock"，因为合并后就无法证明分层是否真的成立。

生产接入点：
  - 牛马仕：把 NiumashiAgent.plan() 换成官方 Agent API/SDK 调用
  - 模型：  MODEL_IMPL=real 即走牛马仕内网算力（Qwen3.5-9B-AWQ），
            失败自动降级确定性 Mock 并打 degraded 标记，绝不 fail-open
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import httpx

from common.config import (
    AGENT_IMPL, EGRESS_ENABLED, HERMES_API_KEY, HERMES_BASE_URL, HERMES_MODEL,
    HERMES_TIMEOUT, MODEL_IMPL,
)
from common.logger import get as get_logger
from common.schema import ACTION_CODES

runlog = get_logger("inner", "model_connector")


@dataclass(frozen=True)
class Endpoint:
    endpoint_id: str
    supplier: str
    version: str
    egress: bool          # True 表示该端点位于外网，调用需经 DMZ 出域
    tier: str
    region: str = "cn-north"      # 数据驻留区域：合规要求"可签发数据等级"必须能落到区域
    auth: str = "api-key"         # 鉴权方式：api-key / oauth2-cc / mtls
    max_data_level: str = "L1"    # 该端点被批准接收的最高数据等级（L0=不出域, L1=脱敏）
    approved_by: str = ""         # 逐端点审批人（空 = 未审批，不得调用）
    effective_from: str = ""      # 审批生效日期


# 端点清单：模型供应商、版本、位置与是否出域，逐项独立登记、独立审批。
# 与 docs/17-模型端点登记表.md 必须逐字一致——文档是给审计看的，代码是执行用的，
# 两者不一致时以"更严格的一方"为准（这正是自检脚本 tools/check_endpoints.py 的作用）。
ENDPOINTS: dict[str, Endpoint] = {
    "ep-internal-a": Endpoint(
        "ep-internal-a", "internal-llm", HERMES_MODEL, False,
        "内网模型 · 牛马仕(Bull Hermes)私有算力",
        region="cn-internal", auth="api-key", max_data_level="L2",
        approved_by="科技+安全（内网端点，随 Agent 部署一并评审）",
        effective_from="2026-11-01"),
    "ep-partner-b": Endpoint(
        "ep-partner-b", "partner-cloud", "llm-b-2.1", True, "外网模型",
        region="cn-east-1", auth="mtls", max_data_level="L1",
        approved_by="安全+数据合规（仅脱敏最小上下文）",
        effective_from="2026-11-01"),
}

# 意图 → 端点选择策略（策略本身是可配置的，不在代码里硬编码业务偏好）
_ROUTING = {
    "DISPUTE_PROCESS": "ep-partner-b",          # 低风险、语义归纳需求高，走外网模型
    "TRANSACTION_QUERY": "ep-partner-b",        # 低风险，可外发脱敏上下文
    "SUSPECTED_UNAUTHORIZED": "ep-internal-a",  # 高风险，模型端点也必须在内网
    "PENDING_CONFIRM": "ep-internal-a",         # 信息不足，避免外发不必要上下文
    "FEEDBACK_REPLY": "ep-internal-a",          # 客户反馈答复：涉客户诉求原文，一律内网
}


class ModelConnector:
    """模型连接器：负责端点选择与调用，不承担任务规划。"""

    name = MODEL_IMPL

    def select(self, intent: str) -> Endpoint:
        if not EGRESS_ENABLED:
            # 气隙/全内网模式：任何意图都不允许选择外网端点
            eid = "ep-internal-a"
        else:
            eid = _ROUTING.get(intent, "ep-internal-a")
        ep = ENDPOINTS.get(eid)
        if ep is None:
            raise RuntimeError(f"模型端点未登记: {eid} —— 未登记端点禁止调用")
        # 逐端点审批闸门：审批人为空 = 该端点未获批准，出域端点一律降级到内网。
        # 这条不是"配置建议"，而是把"未审批端点能被调用"变成不可能。
        if ep.egress and not ep.approved_by:
            runlog.warning("endpoint_not_approved", extra={"extra_data": {
                "endpoint": ep.endpoint_id, "fallback": "ep-internal-a"}})
            return ENDPOINTS["ep-internal-a"]
        return ep

    def invoke_local(self, ep: Endpoint, task: dict) -> dict:
        """内网端点推理。MODEL_IMPL=real 走牛马仕算力，失败降级 Mock（fail-closed）。"""
        if MODEL_IMPL == "real":
            try:
                out = _hermes_completion(task["intent_hint"], task["sanitized_context"], ep)
                runlog.info("hermes_completion_ok", extra={"extra_data": {
                    "intent": task["intent_hint"], "model": ep.version}})
                return out
            except Exception as exc:  # 算力不可达/输出不合规 → 确定性兜底，绝不放空
                runlog.warning("hermes_degraded_to_mock", extra={"extra_data": {
                    "intent": task["intent_hint"],
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}"}})
                out = _mock_completion(task["intent_hint"], task["sanitized_context"], ep)
                out["degraded"] = f"hermes-fallback: {type(exc).__name__}"
                return out
        return _mock_completion(task["intent_hint"], task["sanitized_context"], ep)


_PROMPT_SCHEMA = {
    "customer_reply": "面向客户的最终答复文本：自然、礼貌、易理解，≤150字，禁止内部术语",
    "intent_summary": "内部用：一句话概括客户诉求（≤60字），不下发给客户",
    "reasoning_summary": "内部用：为什么给出该建议（≤80字），不下发给客户",
    "missing_information": "内部用：还缺哪些信息才能处置，数组，可为空",
    "recommended_next_step": "必须是以下动作码之一",
    "risk_flags": "内部用：风险标记数组，涉及资金安全须含 HUMAN_REVIEW_REQUIRED",
    "source_rule_ids": "内部用：引用的内部规则编号数组",
}


def _hermes_completion(intent: str, context: str, ep: Endpoint) -> dict:
    """牛马仕内网算力（vLLM / OpenAI 兼容）。Qwen3 思考模式必须显式关闭，
    否则内容进 reasoning_content、content 为空。输出过不了 schema 校验就抛错降级。"""
    codes = "、".join(sorted(ACTION_CODES))
    sys_prompt = (
        "你是银行的智能客服，直接面向客户撰写回复。只输出一个 JSON 对象，"
        "不要 markdown、不要解释。"
        f"字段要求：{_PROMPT_SCHEMA}。recommended_next_step 只能取：{codes}。"
        "涉资金安全的建议动作用 HUMAN_REVIEW。"
        "撰写 customer_reply 必须遵守以下质量规则："
        "1）只输出面向客户的自然、礼貌、易懂的话，直接回应客户的问题，信息量要足；"
        "2）严禁出现意图摘要、动作码（如 GUIDE_APP_QUERY）、风险标记、字段名等任何内部术语，"
        "不得提及模型/智能体/规则引擎等内部分析过程；"
        "3）没有明确依据时不得编造银行政策、费率、额度、时效；不确定的内容要主动说明"
        "“需以银行最新公告/人工核实为准”；"
        "4）信息不足时先回应已知部分，再围绕本次问题追问，最多问 2 项且必须与处置直接相关，"
        "禁止索要无关信息，不得重复询问客户已提供过的信息；"
        "5）涉及资金安全、账户冻结、交易异常、投诉升级时语气安抚、谨慎处理：提醒客户保管好短信验证码和密码，"
        "并建议 HUMAN_REVIEW 转人工优先跟进；但仅限上述高风险场景——"
        "操作类问题（如登录异常、功能使用、规则咨询）不得转人工，应先安抚、给出可操作的自助排查建议，"
        "再围绕现象追问细节；"
        "6）客户已明确要求人工服务时，直接确认转接，不再劝说使用自助功能；"
        "7）判定口径（v1.5.5，必须严格执行）：下列都属于低风险，一律 recommended_next_step=GUIDE_APP_QUERY，"
        "并在 customer_reply 里给出具体步骤（入口路径 + 需要什么），不得说\"需人工核实\"或\"已转人工\"："
        "① 问办理流程/需要什么材料；② 问在哪个页面查/怎么查（含交易明细、进度）；"
        "③ 问费率、年费、额度、政策在哪看（可答\"以最新公告为准\"）；④ 问能不能改预留手机号、能不能自助办理；"
        "⑤ 问商户编号/扣款是什么（引导按商户号查明细，建议联系商户核实）。"
        "只有出现以下情形才 HUMAN_REVIEW：客户主张资金已损失、要求退款/赔付、要求冻结或挂失、"
        "明确说这笔交易不是本人所为（陌生交易/盗刷）、或要求投诉升级。"
        "拿不准时按低风险处理——先给路径，不要动辄转人工。"
        "8）只回答客户本次问的那一件事：不要附带与本次问题无关的政策、费率或办理说明；"
        "客户问什么就答什么，需要追问最多 1 项，且必须与本次处置直接相关。"
    )
    user_prompt = f"意图分类（内部参考）：{intent}\n脱敏会话上下文：{context}"
    r = httpx.post(
        f"{HERMES_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {HERMES_API_KEY}"},
                json={
            "model": HERMES_MODEL,
            "temperature": 0.2,
            "max_tokens": 320,
            "chat_template_kwargs": {"enable_thinking": False},
            # v1.5.1：vLLM 用 chat_template_kwargs 关思考；OpenAI 兼容端点（如中转站）
            # 不认这个字段，需要 reasoning_effort=none，否则内容落到 reasoning_content、
            # content 为空，schema 校验失败后退化成"转人工"。
            "reasoning_effort": "none",
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=HERMES_TIMEOUT,
    )
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"] or ""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError(f"模型输出无 JSON：{raw[:120]}")
    out = json.loads(m.group(0))
    required = ("intent_summary", "reasoning_summary", "recommended_next_step", "customer_reply")
    if not all(isinstance(out.get(k), str) and out[k].strip() for k in required):
        raise ValueError(f"模型输出缺必要字段：{sorted(out)}")
    if len(out["customer_reply"]) > 300:
        raise ValueError(f"customer_reply 超长：{len(out['customer_reply'])} 字")
    if out["recommended_next_step"] not in ACTION_CODES:
        raise ValueError(f"非法动作码：{out['recommended_next_step']}")
    for k, default in (("missing_information", []), ("risk_flags", []), ("source_rule_ids", [])):
        if not isinstance(out.get(k), list):
            out[k] = default
    out.update(model_supplier=ep.supplier, model_version=ep.version, engine="bull-hermes")
    return out


def _mock_completion(intent: str, context: str, ep: Endpoint) -> dict:
    """确定性 Mock：同一输入必得同一输出，便于回归测试与盲测复现。"""
    base = {
        "model_supplier": ep.supplier,
        "model_version": ep.version,
    }
    if intent == "DISPUTE_PROCESS":
        base.update(
            intent_summary="客户希望了解消费争议的发起渠道及所需材料清单。",
            reasoning_summary="咨询属于流程指引类，无需判断交易真伪；应引导客户自助提交并说明材料要求。",
            missing_information=[],
            recommended_next_step="REQUEST_DISPUTE_FORM",
            risk_flags=[],
            source_rule_ids=["R-DISPUTE-01", "R-DISPUTE-02"],
        )
    elif intent == "TRANSACTION_QUERY":
        base.update(
            intent_summary="客户希望查询一笔交易的处理状态或扣款摘要。",
            reasoning_summary="查询类诉求可通过自助渠道满足；状态口径须以核心系统为准。",
            missing_information=[],
            recommended_next_step="GUIDE_APP_QUERY",
            risk_flags=[],
            source_rule_ids=["R-TXN-01", "R-TXN-02"],
        )
    elif intent == "SUSPECTED_UNAUTHORIZED":
        base.update(
            intent_summary="客户主张一笔交易非本人授权，要求立即处置。",
            reasoning_summary="涉及资金安全与责任认定，超出自动回复范围，须人工核查。",
            missing_information=["交易发生时段", "卡片是否离身", "是否已报警"],
            recommended_next_step="HUMAN_REVIEW",
            risk_flags=["HUMAN_REVIEW_REQUIRED"],
            source_rule_ids=["R-UNAUTH-01", "R-UNAUTH-02"],
        )
    elif intent == "FEEDBACK_REPLY":
        # v1.5.6：Mock 也按客户实际问的事给差异化答复（离线/断网时的演示与回归基线）
        ctx = context or ""
        if any(k in ctx for k in ("争议", "不是我刷", "不是本人")):
            base.update(
                intent_summary="客户咨询交易争议的提交渠道与所需材料。",
                reasoning_summary="流程咨询类，低风险，给出入口与材料清单即可，不承诺时限。",
                missing_information=[], recommended_next_step="REQUEST_DISPUTE_FORM", risk_flags=[],
                source_rule_ids=["R-DISPUTE-01", "R-DISPUTE-02"],
                customer_reply=("您好，交易争议可以在手机银行 App「我的-交易争议」提交，提交时需完成身份核验。"
                                "请准备四项信息：交易时间、交易金额、商户名称、争议原因说明；有交易截图或回单也请一并保留。"))
        elif any(k in ctx for k in ("看不懂", "扣款", "商户", "账单")):
            base.update(
                intent_summary="客户对一笔扣款的商户或用途存疑。",
                reasoning_summary="信息不足，先引导按商户号查明细，并追问时间与金额。",
                missing_information=["交易时间", "交易金额"], recommended_next_step="COLLECT_MORE_INFO", risk_flags=[],
                source_rule_ids=["R-TXN-01", "R-TXN-02"],
                customer_reply=("您好，建议先在手机银行 App「我的-交易明细」里按商户名称或金额找到这笔记录，"
                                "可以看到渠道与商户信息。麻烦补充一下交易时间和大致金额，我帮您进一步核实。"))
        elif any(k in ctx for k in ("冻结", "解冻", "急用钱", "挂失")):
            base.update(
                intent_summary="客户卡片状态异常，要求尽快解除限制。",
                reasoning_summary="涉及账户安全与身份核验，超出自动答复范围，须转人工。",
                missing_information=[], recommended_next_step="HUMAN_REVIEW", risk_flags=["HUMAN_REVIEW_REQUIRED"],
                source_rule_ids=["R-UNAUTH-01", "R-UNAUTH-02"],
                customer_reply=("您好，为保障您的资金安全，这个问题需要人工核实身份后处理，已为您转接人工客服优先跟进，"
                                "请保持电话畅通。请不要向任何人透露短信验证码和密码。"))
        elif any(k in ctx for k in ("年费", "费率", "免年费")):
            base.update(
                intent_summary="客户咨询信用卡年费政策与减免条件。",
                reasoning_summary="政策咨询类，指引官方查询入口，不承诺减免。",
                missing_information=[], recommended_next_step="GUIDE_APP_QUERY", risk_flags=[],
                source_rule_ids=["R-TXN-01"],
                customer_reply=("您好，年费政策与卡种和用卡达标情况有关，可在手机银行 App「信用卡-我的卡片-年费查询」"
                                "或官网费率公告查看，具体以银行最新公告为准。"))
        else:
            base.update(
                intent_summary="客户提交了业务反馈，需给出指引或转人工。",
                reasoning_summary="按固定口径指引自助渠道办理，不承诺任何业务结果。",
                missing_information=[], recommended_next_step="GUIDE_APP_QUERY", risk_flags=[],
                source_rule_ids=["R-FB-01"],
                customer_reply=("您好，感谢您的反馈，您反映的问题我们已记录并安排跟进。"
                                "相关业务可以先在手机银行 App 对应功能页面自助查询或办理。"
                                "如未能解决，回复补充具体情况，我会继续为您核实；需要人工可直接回复“转人工”。"))
    else:  # PENDING_CONFIRM：模型给出倾向性建议，但动作与规则冲突，用于演示"规则优先"
        base.update(
            intent_summary="客户诉求表述不明确，可能同时涉及查询与争议。",
            reasoning_summary="信息不足，推测客户主要想确认扣款来源，建议先引导自助查询。",
            missing_information=["具体交易时间", "诉求类型（查询/争议）"],
            recommended_next_step="GUIDE_APP_QUERY",
            risk_flags=[],
            source_rule_ids=["R-PENDING-01"],
        )
    return base


class NiumashiAgent:
    """微光「牛马仕」(Bull Hermes) 智能体：负责任务理解、规划、模型选择与结果组织。

    它是一个 Agent，不是模型。自身不做推理，只编排。
    """

    version = "nms-agent-0.9.3"

    def __init__(self) -> None:
        self.connector = ModelConnector()

    def plan(self, intent: str, sanitized_context: str) -> dict:
        if AGENT_IMPL != "mock":
            raise NotImplementedError("AGENT_IMPL 仅提供 mock 实现，生产请接入牛马仕官方 Agent API/SDK")

        ep = self.connector.select(intent)
        task = {"intent_hint": intent, "sanitized_context": sanitized_context}
        plan = {
            "agent_version": self.version,
            "endpoint": ep.endpoint_id,
            "model_supplier": ep.supplier,
            "model_version": ep.version,
            "route": "EXTERNAL" if ep.egress else "INTERNAL",
            "steps": [
                "解析意图与约束",
                f"选择模型端点 {ep.endpoint_id}",
                "生成结构化候选建议",
                "标注风险标记与规则引用",
            ],
        }
        if not ep.egress:
            plan["output"] = self.connector.invoke_local(ep, task)
        else:
            # 外网端点：输出由外网模型产生，本方法只返回路由决定
            plan["task"] = task
        return plan
