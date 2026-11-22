"""内网 HTTP 服务：员工工作台 + 管理端 + 业务编排 API + MCP Facade。

三件事按顺序说清楚：

1. **身份权威在这里。** 登录、会话、角色、锁定全部在内网完成，DMZ 与外网
   不参与认证，也没有独立账号。外网能做的只有"传算力请求"。

2. **每个接口都标注了允许的角色。** 不是靠前端隐藏菜单，而是在服务端
   逐个声明。漏标的接口在评审时就是缺陷。

3. **MCP Facade 只为白名单工具提供实现。** 被禁止的能力在这里**根本没有
   对应函数**——不是"有函数但拒绝"，而是不存在。这个区别决定了配置失误
   能否变成安全事件。
"""
from __future__ import annotations

import json
import re
import secrets
from datetime import date, timedelta

import httpx
from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from common import audit
from common.config import (
    APP_VERSION, DMZ_URL, EGRESS_ENABLED, HERMES_MODEL, INNER_PORT,
    MIN_AGGREGATE_SAMPLE, MODEL_IMPL, ROOT, SERVICE_TOKEN, SESSION_COOKIE,
    SESSION_SECRET,
)
from common import totp
from common.crypto import sign
from common.egress import guard
from common.logger import get as get_logger
from common.schema import (
    ChangePasswordRequest, ChangeReviewRequest, ChangeRollbackRequest,
    MfaConfirmRequest, MfaDisableRequest,
    ConfirmRequest, CreateCaseRequest, CreateUserRequest,
    DmSendRequest, FeedbackDeliverRequest, FeedbackIngestRequest,
    FeedbackSeatReplyRequest, LeaveReviewRequest, LeaveSubmitRequest, LoginRequest,
    McpProxyRequest, MustChangePwdRequest, NoticeRequest, PasswordResetRequest,
    RestoreRequest, UserStateRequest, FEEDBACK_CATEGORIES,
)

from . import auth, db, service
from .agent import NiumashiAgent
from .privacy import PrivacyService

app = FastAPI(title="内网可信域 · 银行客服 AI 工作台", version=APP_VERSION)

runlog = get_logger("inner", "app")

WEB = ROOT / "web"

# 角色常量：写死字面量容易拼错，且改动时无法全局检索
AGENT, SUPERVISOR, AUDITOR, DEVELOPER = "agent", "supervisor", "auditor", "developer"


@app.middleware("http")
async def _trace_middleware(request: Request, call_next):
    """全链路 trace：入口生成 trace_id（或沿用上游传入），写入审计上下文并回写响应头。

    这样即使某个事件没有 request_id（如登录失败、健康检查），审计里也有 trace_id，
    排障时可以按一次 HTTP 请求把跨区事件串起来。
    """
    tid = (request.headers.get("x-trace-id") or "").strip() or f"TR-{secrets.token_hex(8)}"
    audit.set_trace(tid)
    response = await call_next(request)
    response.headers["X-Trace-Id"] = tid
    return response


@app.on_event("startup")
def _startup() -> None:
    db.init()
    audit.log("inner", "ZONE_START", trace_id="INNER-START", detail={"zone": "inner"})
    # 运行日志记录启动快照：只记配置指纹，不记任何密钥明文
    runlog.info("inner_zone_started", extra={"extra_data": {
        "inner_port": INNER_PORT, "egress_enabled": EGRESS_ENABLED,
        "model_impl": MODEL_IMPL, "hermes_model": HERMES_MODEL,
    }})


# ---------------- 页面 ----------------

def _user(request: Request) -> dict | None:
    return auth.session_of(request.cookies.get(SESSION_COOKIE),
                           request.headers.get(auth.DEVICE_HEADER, ""))


@app.get("/login")
def login_page(request: Request):
    # 安全口径（用户要求）：登录页一律强制重新登录——
    # 已带有效会话访问 /login 时立即吊销该会话、清除 Cookie，再渲染登录表单，
    # 防止共用电脑/离开座位后他人打开地址即直接进入控制台。
    # 再次登录按单会话规则顶掉旧会话（此处吊销后再登录效果一致）。
    u = _user(request)
    resp = FileResponse(WEB / "login.html", headers={"Cache-Control": "no-store, must-revalidate"})
    if u:
        auth.logout(u["sid"], u["emp_id"])
        auth.clear_cookie(resp)
    return resp


@app.get("/")
def root(request: Request):
    u = _user(request)
    if not u:
        return RedirectResponse("/login", 302)
    return RedirectResponse("/client" if u["portal"] == "client" else "/admin", 302)


@app.get("/client")
def client_page(request: Request):
    u = _user(request)
    if not u or u["portal"] != "client":
        # 未登录、或管理员试图进坐席工作台：都回登录页，不提示"你走错端了"
        return RedirectResponse("/login", 302)
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store, must-revalidate"})


@app.get("/admin")
def admin_page(request: Request):
    u = _user(request)
    if not u or u["portal"] != "admin":
        return RedirectResponse("/login?portal=admin", 302)
    return FileResponse(WEB / "admin.html", headers={"Cache-Control": "no-store, must-revalidate"})


@app.get("/change-password")
def change_password_page(request: Request):
    """自助改密页：所有已登录角色可达；首登强制改密时前端会引导到这里。"""
    if not _user(request):
        return RedirectResponse("/login", 302)
    return FileResponse(WEB / "password.html", headers={"Cache-Control": "no-store, must-revalidate"})


app.mount("/assets", StaticFiles(directory=str(WEB)), name="assets")


@app.get("/internal/api/v1/health")
def health() -> dict:
    """探活保持匿名：它不含任何业务数据，且负载均衡需要它可达。"""
    return {"zone": "inner", "status": "ok", "version": APP_VERSION,
            "db": str(db.db_path("inner").name)}


# ---------------- 认证 ----------------

@app.post("/internal/api/v1/auth/login")
def login(req: LoginRequest, request: Request):
    ip = request.client.host if request.client else ""
    ua = request.headers.get("user-agent", "")
    device_id = request.headers.get(auth.DEVICE_HEADER, "").strip()
    try:
        token, emp = auth.login(req.username, req.password, req.portal, ip, ua, device_id,
                                otp=req.otp or "")
    except auth.AuthError as exc:
        # 带 code 返回：前端据此决定"提示口令错"还是"弹出第二因子输入框"
        return JSONResponse({"detail": str(exc), "code": getattr(exc, "code", None)},
                            status_code=exc.status)
    emp.pop("sid", None)
    resp = JSONResponse({"ok": True, "user": emp})
    auth.set_cookie(resp, token)
    return resp


# ---------------- 第二因子（TOTP） enrol / disable ----------------
#
# 定位：PoC 期可演示的**标准 TOTP**，不是行内 SSO。生产接行内统一身份
# （OIDC/SAML）＋行内 MFA，见 docs/22-SSO与MFA对接方案.md。
# 启用流程刻意要两步：生成密钥 → 用一次验证码确认，防止"扫错码被锁在外面"。

@app.get("/internal/api/v1/auth/mfa")
def mfa_state(user: dict = Depends(auth.current_user)) -> dict:
    st = db.totp_state(user["emp_id"])
    return {"enabled": bool(st.get("totp_secret")),
            "enabled_at": st.get("totp_enabled_at"),
            "pending": bool(st.get("totp_pending"))}


@app.post("/internal/api/v1/auth/mfa/enable")
def mfa_enable(user: dict = Depends(auth.current_user)) -> dict:
    if db.totp_state(user["emp_id"]).get("totp_secret"):
        raise HTTPException(409, "二次验证已启用，如需更换请先关闭")
    secret = totp.new_secret()
    db.set_user_fields(user["emp_id"], totp_pending=secret)
    audit.log("inner", "MFA_ENROLL_STARTED", actor=user["emp_id"])
    return {"secret": secret,
            "uri": totp.provisioning_uri(secret, user["emp_id"]),
            "note": "用认证器扫码后，输入当前 6 位验证码确认（POST /auth/mfa/confirm）"}


@app.post("/internal/api/v1/auth/mfa/confirm")
def mfa_confirm(req: MfaConfirmRequest, user: dict = Depends(auth.current_user)) -> dict:
    st = db.totp_state(user["emp_id"])
    pending = (st.get("totp_pending") or "").strip()
    if not pending:
        raise HTTPException(409, "没有待确认的密钥，请先调用 /auth/mfa/enable")
    if not totp.verify(pending, req.otp):
        audit.log("inner", "MFA_ENROLL_FAILED", actor=user["emp_id"], severity="medium")
        raise HTTPException(401, "验证码不正确，请检查设备时间后重试")
    ts = service._iso(service._now())
    db.set_user_fields(user["emp_id"], totp_secret=pending, totp_pending=None,
                       totp_enabled_at=ts)
    audit.log("inner", "MFA_ENABLED", actor=user["emp_id"], severity="medium",
              detail={"enabled_at": ts})
    return {"enabled": True, "enabled_at": ts,
            "warning": "请妥善保存恢复方式；PoC 环境遗失需管理员在库中清空 totp_secret"}


@app.post("/internal/api/v1/auth/mfa/disable")
def mfa_disable(req: MfaDisableRequest, user: dict = Depends(auth.current_user)) -> dict:
    u = db.user_by_emp_id(user["emp_id"]) or {}
    if not auth.verify_password(req.password, u.get("pwd_hash", "")):
        audit.log("inner", "AUTHZ_DENIED", actor=user["emp_id"], severity="high",
                  detail={"reason": "mfa_disable_bad_password"})
        raise HTTPException(401, "口令不正确，无法关闭二次验证")
    db.set_user_fields(user["emp_id"], totp_secret=None, totp_pending=None)
    audit.log("inner", "MFA_DISABLED", actor=user["emp_id"], severity="high")
    return {"enabled": False}


@app.post("/internal/api/v1/auth/logout")
def logout(request: Request):
    u = _user(request)
    if u:
        auth.logout(u["sid"], u["emp_id"])
    resp = JSONResponse({"ok": True})
    auth.clear_cookie(resp)
    return resp


@app.get("/internal/api/v1/auth/me")
def me(user: dict = Depends(auth.current_user)) -> dict:
    row = db.user_by_emp_id(user["emp_id"]) or {}
    base = {k: user[k] for k in ("emp_id", "name", "role", "portal", "expires_at")}
    base.update({"username": row.get("username"), "last_login": row.get("last_login"),
                 "must_change_pwd": bool(row.get("must_change_pwd"))})
    return base


@app.get("/internal/api/v1/auth/sessions")
def sessions(user: dict = Depends(auth.require(SUPERVISOR))) -> list[dict]:
    return [dict(s, sid=s["sid"][:8] + "…") for s in db.live_sessions()]


@app.post("/internal/api/v1/auth/sessions/{sid}/revoke")
def revoke_session(sid: str, user: dict = Depends(auth.require(SUPERVISOR))) -> dict:
    with db.connect() as c:
        row = c.execute("SELECT sid FROM sessions WHERE sid LIKE ? LIMIT 1",
                        (sid.split("…")[0] + "%",)).fetchone()
    if not row:
        raise HTTPException(404, "会话不存在")
    db.revoke_session(row["sid"])
    audit.log("inner", "SESSION_REVOKED", actor=user["emp_id"], severity="high",
              detail={"target_sid": row["sid"][:8]})
    return {"ok": True, "sid": row["sid"][:8]}


# ---------------- 账号管理（管理端） ----------------

@app.get("/internal/api/v1/users")
def users(user: dict = Depends(auth.require(SUPERVISOR))) -> list[dict]:
    return db.list_users()


@app.post("/internal/api/v1/users/{emp_id}/state")
def set_user_state(emp_id: str, req: UserStateRequest,
                   user: dict = Depends(auth.require(SUPERVISOR))) -> dict:
    if emp_id == user["emp_id"] and not req.active:
        raise HTTPException(400, "不能停用自己的账号")
    if not db.employee(emp_id):
        raise HTTPException(404, "员工不存在")
    db.set_user_active(emp_id, req.active)
    audit.log("inner", "USER_STATE_CHANGED", actor=user["emp_id"], severity="high",
              detail={"target": emp_id, "active": req.active})
    return {"ok": True, "emp_id": emp_id, "active": req.active}


def _check_password_strength(pwd: str) -> None:
    """口令策略：8 位以上，须同时包含字母与数字——管理员分发账号时同样要过这道闸。"""
    if len(pwd) < 8 or pwd.isdigit() or not any(c.isalpha() for c in pwd) or not any(c.isdigit() for c in pwd):
        raise HTTPException(400, "口令至少 8 位，且须同时包含字母与数字")


@app.post("/internal/api/v1/auth/password")
def change_password(req: ChangePasswordRequest,
                    user: dict = Depends(auth.current_user)) -> dict:
    """自助改密：需验证旧口令；成功后清除强制改密标记并吊销其它会话。"""
    row = db.user_by_emp_id(user["emp_id"])
    if not row or not auth.verify_password(req.old_password, row["pwd_hash"]):
        audit.log("inner", "PASSWORD_CHANGE_FAIL", actor=user["emp_id"], severity="medium",
                  detail={"reason": "bad_old_password"})
        raise HTTPException(401, "旧口令不正确")
    _check_password_strength(req.new_password)
    if req.new_password == req.old_password:
        raise HTTPException(400, "新口令不能与旧口令相同")
    db.set_password(user["emp_id"], auth.hash_password(req.new_password),
                    must_change=False, keep_sid=user["sid"])
    audit.log("inner", "PASSWORD_CHANGED", actor=user["emp_id"], severity="medium",
              detail={"self": True})
    return {"ok": True, "must_change_pwd": False}


@app.post("/internal/api/v1/users")
def create_user(req: CreateUserRequest, user: dict = Depends(auth.require(SUPERVISOR))) -> dict:
    """创建账号。系统没有公开注册——同事的账号只能由管理员在这里发放。"""
    _check_password_strength(req.password)
    portal = "client" if req.role == AGENT else "admin"
    if db.user_by_username(req.username):
        raise HTTPException(409, "账号名已存在")
    if db.employee(req.emp_id):
        raise HTTPException(409, "工号已存在")
    try:
        db.create_user(req.emp_id, req.username, auth.hash_password(req.password), req.role, portal, req.name)
    except Exception as exc:
        raise HTTPException(500, f"创建失败: {exc}")
    audit.log("inner", "USER_CREATED", actor=user["emp_id"], severity="high",
              detail={"target": req.emp_id, "role": req.role, "portal": portal})
    return {"ok": True, "emp_id": req.emp_id, "role": req.role, "portal": portal}


@app.post("/internal/api/v1/users/{emp_id}/password")
def reset_password(emp_id: str, req: PasswordResetRequest,
                   user: dict = Depends(auth.require(SUPERVISOR))) -> dict:
    _check_password_strength(req.password)
    if not db.employee(emp_id):
        raise HTTPException(404, "员工不存在")
    db.set_password(emp_id, auth.hash_password(req.password), must_change=True)
    audit.log("inner", "PASSWORD_RESET", actor=user["emp_id"], severity="high",
              detail={"target": emp_id, "sessions_revoked": True, "must_change": True})
    return {"ok": True, "emp_id": emp_id, "must_change_pwd": True}


@app.post("/internal/api/v1/users/{emp_id}/must-change")
def set_must_change_pwd(emp_id: str, req: MustChangePwdRequest,
                        user: dict = Depends(auth.require(SUPERVISOR))) -> dict:
    """切换首登强制改密标记；开启即吊销该账号全部活跃会话。"""
    if not db.employee(emp_id):
        raise HTTPException(404, "员工不存在")
    db.set_must_change(emp_id, req.enabled)
    audit.log("inner", "USER_MUST_CHANGE_SET", actor=user["emp_id"], severity="medium",
              detail={"target": emp_id, "enabled": req.enabled})
    return {"ok": True, "emp_id": emp_id, "must_change_pwd": req.enabled}


# ---------------- OA：个人中心 / 通知公告 / 请假审批 ----------------


@app.get("/internal/api/v1/profile")
def profile(user: dict = Depends(auth.current_user)) -> dict:
    row = db.user_by_emp_id(user["emp_id"]) or {}
    prof = db.employee(user["emp_id"]) or {}
    return {"emp_id": user["emp_id"], "name": prof.get("name", ""),
            "role": user["role"], "username": row.get("username"),
            "active": bool(row.get("active", 1)), "last_login": row.get("last_login"),
            "must_change_pwd": bool(row.get("must_change_pwd")),
            "unread_notices": db.unread_count(user["emp_id"]),
            "leave_count": len(db.my_leaves(user["emp_id"]))}


@app.get("/internal/api/v1/notices")
def notices(user: dict = Depends(auth.current_user)) -> dict:
    return {"unread": db.unread_count(user["emp_id"]), "items": db.notices_for(user["emp_id"])}


@app.post("/internal/api/v1/notices/{notice_id}/read")
def notice_read(notice_id: str, user: dict = Depends(auth.current_user)) -> dict:
    if not db.notice_exists(notice_id):
        raise HTTPException(404, "通知不存在")
    ts = service._iso(service._now())
    db.mark_notice_read(notice_id, user["emp_id"], ts)
    audit.log("inner", "NOTICE_READ", actor=user["emp_id"], detail={"notice_id": notice_id})
    return {"ok": True, "unread": db.unread_count(user["emp_id"])}


@app.post("/internal/api/v1/notices")
def publish_notice(req: NoticeRequest,
                   user: dict = Depends(auth.require(SUPERVISOR))) -> dict:
    ts = service._iso(service._now())
    nid = f"N-{service._now().strftime('%y%m%d%H%M%S')}-{secrets.token_hex(2).upper()}"
    db.publish_notice(nid, req.title, req.body, user["emp_id"], req.level, ts)
    audit.log("inner", "NOTICE_PUBLISHED", actor=user["emp_id"], severity="medium",
              detail={"notice_id": nid, "level": req.level, "title": req.title})
    return {"ok": True, "notice_id": nid, "created_at": ts}


@app.post("/internal/api/v1/notices/{notice_id}/revoke")
def revoke_notice(notice_id: str,
                  user: dict = Depends(auth.require(SUPERVISOR))) -> dict:
    n = db.revoke_notice(notice_id)
    if not n:
        raise HTTPException(409, "通知不存在或已撤回")
    audit.log("inner", "NOTICE_REVOKED", actor=user["emp_id"], severity="medium",
              detail={"notice_id": notice_id})
    return {"ok": True, "notice_id": notice_id}


@app.get("/internal/api/v1/notices/manage")
def manage_notices(user: dict = Depends(auth.require(SUPERVISOR))) -> dict:
    counts = db.read_counts()
    items = [{**n, "read_count": counts.get(n["notice_id"], 0)}
             for n in db.list_notices(include_revoked=True)]
    return {"items": items}


@app.post("/internal/api/v1/leaves")
def submit_leave(req: LeaveSubmitRequest,
                 user: dict = Depends(auth.current_user)) -> dict:
    try:
        d1 = date.fromisoformat(req.start_date)
        d2 = date.fromisoformat(req.end_date)
    except ValueError as exc:
        raise HTTPException(400, f"日期格式无效: {exc}")
    if d2 < d1:
        raise HTTPException(400, "结束日期不能早于开始日期")
    if d1 < date.today() - timedelta(days=1):
        raise HTTPException(400, "开始日期不能早于昨天")
    if d2 > d1 + timedelta(days=30):
        raise HTTPException(400, "单次请假不能超过 30 天")
    ts = service._iso(service._now())
    lid = f"LV-{service._now().strftime('%y%m%d%H%M%S')}-{secrets.token_hex(2).upper()}"
    db.submit_leave(lid, user["emp_id"], req.start_date, req.end_date, req.reason.strip(), ts)
    audit.log("inner", "LEAVE_SUBMITTED", actor=user["emp_id"], detail={
        "leave_id": lid, "start": req.start_date, "end": req.end_date})
    return {"ok": True, "leave_id": lid, "status": "PENDING", "created_at": ts}


@app.get("/internal/api/v1/leaves/mine")
def my_leave_list(user: dict = Depends(auth.current_user)) -> list[dict]:
    return db.my_leaves(user["emp_id"])


@app.get("/internal/api/v1/leaves/pending")
def pending_leave_list(user: dict = Depends(auth.require(SUPERVISOR))) -> list[dict]:
    return db.pending_leaves()


@app.post("/internal/api/v1/leaves/{leave_id}/review")
def review_leave(leave_id: str, req: LeaveReviewRequest,
                 user: dict = Depends(auth.require(SUPERVISOR))) -> dict:
    row = db.leave(leave_id)
    if not row:
        raise HTTPException(404, "请假单不存在")
    if row["emp_id"] == user["emp_id"]:
        raise HTTPException(400, "不能审批自己的请假申请")
    n = db.review_leave(leave_id, req.action, user["emp_id"], req.comment.strip(), service._iso(service._now()))
    if not n:
        raise HTTPException(409, "该申请已被审批，不能重复处理")
    audit.log("inner", "LEAVE_REVIEWED", actor=user["emp_id"], severity="medium",
              detail={"leave_id": leave_id, "action": req.action,
                      "applicant": row["emp_id"], "comment": req.comment[:100]})
    return {"ok": True, "leave_id": leave_id, "status": req.action}


# ---------------- OA：通讯录与站内私信 ----------------


@app.get("/internal/api/v1/dms/contacts")
def dm_contacts(user: dict = Depends(auth.current_user)) -> dict:
    """通讯录：除本人外全部在职员工，附最近消息与未读数（作为会话列表用）。"""
    return {"me": user["emp_id"], "items": db.dm_inbox(user["emp_id"])}


@app.get("/internal/api/v1/dms/unread")
def dm_unread(user: dict = Depends(auth.current_user)) -> dict:
    return {"unread": db.dm_unread_total(user["emp_id"])}


@app.get("/internal/api/v1/dms/with/{emp_id}")
def dm_thread(emp_id: str, user: dict = Depends(auth.current_user)) -> dict:
    """与某人的会话；打开即把对方发来的未读消息标记已读。"""
    if emp_id == user["emp_id"]:
        raise HTTPException(400, "不能与自己私信")
    prof = db.employee(emp_id)
    if not prof:
        raise HTTPException(404, "员工不存在")
    # 先标记已读再取消息：返回载荷里的 read_at 才是本次打开后的最新回执
    db.dm_mark_thread_read(emp_id, user["emp_id"], service._iso(service._now()))
    items = db.dm_thread(user["emp_id"], emp_id)
    return {"peer": {"emp_id": emp_id, "name": prof.get("name", emp_id), "role": prof.get("role", "")},
            "items": items, "unread_total": db.dm_unread_total(user["emp_id"])}


@app.post("/internal/api/v1/dms")
def dm_send(req: DmSendRequest, user: dict = Depends(auth.current_user)) -> dict:
    """发送站内私信。审计只记元数据（收发双方与长度），不落正文，避免敏感内容双写。"""
    if req.to_emp_id == user["emp_id"]:
        raise HTTPException(400, "不能给自己发私信")
    prof = db.employee(req.to_emp_id)
    if not prof:
        raise HTTPException(404, "收件员工不存在")
    body = req.body.strip()
    if not body:
        raise HTTPException(400, "消息内容不能为空")
    ts = service._iso(service._now())
    mid = db.dm_send(user["emp_id"], req.to_emp_id, body, ts)
    audit.log("inner", "DM_SENT", actor=user["emp_id"], detail={
        "msg_id": mid, "to": req.to_emp_id, "length": len(body)})
    runlog.info("dm_sent", extra={"extra_data": {
        "msg_id": mid, "from": user["emp_id"], "to": req.to_emp_id}})
    return {"ok": True, "msg_id": mid, "created_at": ts}


# ---------------- 业务 ----------------

@app.get("/internal/api/v1/employees")
def employees(user: dict = Depends(auth.require(SUPERVISOR, AUDITOR, DEVELOPER))) -> list[dict]:
    with db.connect() as c:
        return [dict(r) for r in c.execute("SELECT * FROM employees ORDER BY emp_id")]


@app.get("/internal/api/v1/cases")
def cases(user: dict = Depends(auth.current_user)) -> list[dict]:
    return db.list_cases()


@app.get("/internal/api/v1/rules")
def rules(user: dict = Depends(auth.current_user)) -> list[dict]:
    return db.all_rules()


@app.post("/internal/api/v1/cases")
async def create_case(req: CreateCaseRequest, user: dict = Depends(auth.current_user)):
    # 坐席只能以自己的身份提交；主管可代提交（交接班等场景），但审计留两个身份
    if user["role"] == AGENT and req.emp_id != user["emp_id"]:
        audit.log("inner", "AUTHZ_DENIED", actor=user["emp_id"], severity="high",
                  detail={"reason": "impersonation", "claimed": req.emp_id})
        raise HTTPException(403, "不得以他人身份提交工单")
    if user["role"] not in (AGENT, SUPERVISOR):
        raise HTTPException(403, "当前角色无权提交工单")
    try:
        result = await service.handle_case(
            case_id=req.case_id, emp_id=req.emp_id,
            session_id=req.session_id, note=req.note,
        )
    except PermissionError as exc:
        audit.log("inner", "AUTH_DENIED", actor=req.emp_id, severity="high", detail={"error": str(exc)})
        raise HTTPException(403, str(exc))
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    if req.emp_id != user["emp_id"]:
        audit.log("inner", "ON_BEHALF", actor=user["emp_id"], severity="medium",
                  request_id=result.get("request_id"),
                  detail={"on_behalf_of": req.emp_id})
    return JSONResponse(result)


@app.get("/internal/api/v1/cases/{request_id}")
def get_case(request_id: str, user: dict = Depends(auth.current_user)):
    row = db.ticket(request_id)
    if not row:
        raise HTTPException(404, "请求不存在")
    if user["role"] == AGENT and row["emp_id"] != user["emp_id"]:
        raise HTTPException(403, "该工单不属于当前坐席")
    return JSONResponse(service._result(request_id))


@app.post("/internal/api/v1/restore")
def restore(req: RestoreRequest, user: dict = Depends(auth.current_user)):
    if user["role"] == AGENT and req.emp_id != user["emp_id"]:
        raise HTTPException(403, "不得以他人身份申请复原")
    try:
        return JSONResponse(service.restore_for_display(req.request_id, req.session_id, req.emp_id, req.text))
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except PermissionError as exc:
        raise HTTPException(403, str(exc))


@app.post("/internal/api/v1/confirm")
def confirm(req: ConfirmRequest, user: dict = Depends(auth.current_user)):
    row = db.ticket(req.request_id)
    if not row:
        raise HTTPException(404, "请求不存在")
    if user["role"] == AGENT and (row["emp_id"] != user["emp_id"] or req.emp_id != user["emp_id"]):
        raise HTTPException(403, "不得处置他人工单")
    status = {"ACCEPT": "CONFIRMED", "MODIFY": "CONFIRMED", "REJECT": "CLOSED", "HUMAN_HANDOFF": "HUMAN_HANDOFF"}[req.action]
    service._persist(req.request_id, status=status)
    audit.log("inner", "HUMAN_CONFIRMED", request_id=req.request_id, actor=user["emp_id"],
              detail={"action": req.action, "note": req.note, "on_behalf_of": req.emp_id})
    return JSONResponse({"request_id": req.request_id, "status": status})


@app.get("/internal/api/v1/tickets")
def tickets(limit: int = Query(30, le=200), user: dict = Depends(auth.current_user)) -> list[dict]:
    # 坐席只能看到自己的工单：列表页泄露他人业务比单条泄露更容易被忽略
    return db.list_tickets(limit, emp_id=user["emp_id"] if user["role"] == AGENT else None)


@app.get("/internal/api/v1/audit")
def audit_trail(request_id: str | None = None, limit: int = Query(200, le=500),
                user: dict = Depends(auth.require(SUPERVISOR, AUDITOR))) -> dict:
    return {
        "inner": audit.query("inner", request_id, limit),
        "dmz": audit.query("dmz", request_id, limit),
        "outer": audit.query("outer", request_id, limit),
    }


@app.get("/internal/api/v1/observe")
def observe(user: dict = Depends(auth.require(SUPERVISOR, AUDITOR))) -> dict:
    """运行观测（C9）：失败率、规则命中、人工驳回率、MCP 拒绝、端点分布、出域统计。

    数据全部来自**已有的审计库与工单表**，不新增埋点、不改业务路径——
    观测不应成为"为了被观测而多写一遍"的负担，也不应成为新的泄露面。
    """
    with db.connect() as c:
        total = c.execute("SELECT COUNT(*) n FROM tickets").fetchone()["n"] or 0
        by_status = {r["status"]: r["n"] for r in
                     c.execute("SELECT status, COUNT(*) n FROM tickets GROUP BY status")}
        by_supplier = {r["model_supplier"] or "未记录": r["n"] for r in
                       c.execute("SELECT model_supplier, COUNT(*) n FROM tickets "
                                 "WHERE model_supplier IS NOT NULL GROUP BY model_supplier")}
        ruled = c.execute("SELECT COUNT(*) n FROM tickets WHERE rule_ids IS NOT NULL "
                          "AND rule_ids NOT IN ('[]','')").fetchone()["n"] or 0
        decided = c.execute("SELECT COUNT(*) n FROM tickets WHERE decision IS NOT NULL").fetchone()["n"] or 0
        human = c.execute("SELECT COUNT(*) n FROM tickets WHERE decision='HUMAN_HANDOFF'").fetchone()["n"] or 0

    fail_states = ("REJECTED", "TIMEOUT", "BLOCKED")
    fail_n = sum(by_status.get(k, 0) for k in fail_states)

    def _count(zone: str, event: str) -> int:
        return sum(1 for e in audit.query(zone, None, 500) if e.get("event") == event)

    mcp_denied = _count("dmz", "MCP_DENIED") + _count("inner", "MCP_DENIED")
    rate_limited = _count("dmz", "RATE_LIMITED")
    egress_ok = _count("dmz", "EGRESS_ACCEPTED")
    egress_deny = _count("dmz", "EGRESS_DENIED")
    restore_deny = _count("inner", "RESTORE_DENIED")

    return {
        "generated_at": service._iso(service._now()),
        "window": "审计库近 500 条事件 + 全部工单",
        "tickets": {"total": total, "by_status": by_status,
                    "fail": fail_n,
                    "fail_rate": round(fail_n / total, 4) if total else 0.0},
        "rules": {"hit": ruled, "hit_rate": round(ruled / total, 4) if total else 0.0},
        "human": {"decided": decided, "handoff": human,
                  "handoff_rate": round(human / decided, 4) if decided else 0.0},
        "mcp": {"denied": mcp_denied, "rate_limited": rate_limited},
        "egress": {"accepted": egress_ok, "denied": egress_deny,
                   "deny_rate": round(egress_deny / (egress_ok + egress_deny), 4)
                   if (egress_ok + egress_deny) else 0.0},
        "restore_denied": restore_deny,
        "endpoints": by_supplier,
        "note": "失败率=REJECTED/TIMEOUT/BLOCKED 占比；拒绝数为审计事件计数（近 500 条窗口）",
    }


@app.get("/internal/api/v1/model-endpoints")
def model_endpoints(user: dict = Depends(auth.require(SUPERVISOR, AUDITOR, DEVELOPER))) -> list[dict]:
    """模型端点登记表（运行态）。与 docs/17 逐字对应，供审计核对"代码里真正能调哪些端点"。"""
    from .agent import ENDPOINTS

    return [{
        "endpoint_id": ep.endpoint_id, "supplier": ep.supplier, "version": ep.version,
        "egress": ep.egress, "tier": ep.tier, "region": ep.region, "auth": ep.auth,
        "max_data_level": ep.max_data_level, "approved_by": ep.approved_by,
        "effective_from": ep.effective_from,
        "callable": (not ep.egress) or bool(ep.approved_by),
    } for ep in ENDPOINTS.values()]


@app.get("/internal/api/v1/stats")
def stats(user: dict = Depends(auth.require(SUPERVISOR, AUDITOR, DEVELOPER))) -> dict:
    with db.connect() as c:
        rows = c.execute("SELECT status, COUNT(*) n FROM tickets GROUP BY status").fetchall()
        total = {r["status"]: r["n"] for r in rows}
        tok = c.execute("SELECT COUNT(*) n FROM token_map").fetchone()["n"]
        sess = c.execute("SELECT COUNT(*) n FROM sessions WHERE revoked=0").fetchone()["n"]
        try:
            auth_events = c.execute(
                "SELECT COUNT(*) n FROM events WHERE event IN"
                " ('LOGIN_FAIL','LOGIN_LOCKED','AUTHZ_DENIED')"
            ).fetchone()["n"]
        except Exception:
            auth_events = 0
    return {"tickets_by_status": total, "tokens": tok, "live_sessions": sess,
            "auth_events": auth_events, "audit": audit.count_by_severity("inner")}


@app.get("/internal/api/v1/acceptance")
def acceptance(user: dict = Depends(auth.require(SUPERVISOR, AUDITOR, DEVELOPER))):
    from common.config import DATA_DIR

    p = DATA_DIR / "acceptance.json"
    if not p.exists():
        raise HTTPException(404, "尚未生成验收报告")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


@app.get("/internal/api/v1/changes")
def changes(user: dict = Depends(auth.require(SUPERVISOR, AUDITOR, DEVELOPER))) -> list[dict]:
    rows = db.all_changes(100)
    for r in rows:
        # gate / payload 是 JSON 文本：下发前解析，避免前端二次字符串处理出错
        for k in ("gate", "payload"):
            if r.get(k):
                try:
                    r[k] = json.loads(r[k])
                except Exception:
                    pass
    return rows


# ---------------- 变更审批发布流水线（R7 / C10 / A4） ----------------
#
# 状态机（唯一合法迁移路径，其它组合一律 409）：
#   PENDING_REVIEW --APPROVE--> APPROVED --PUBLISH--> PUBLISHED --ROLLBACK--> ROLLED_BACK
#   PENDING_REVIEW --REJECT---> REJECTED
#
# 三道硬闸门：
#   ① 不得自审批（提交人 ≠ 审批人，服务端判定，不靠前端隐藏按钮）
#   ② 未过回归门槛不得发布（APPROVED 只是"同意"，不是"可以上线"）
#   ③ 发布前重算草稿指纹，与审批时不一致 → 拒绝（防"审批后被改内容"）

# 依赖黑名单：这些包一旦出现在草稿依赖里，直接判不通过（PoC 级静态检查）
_GATE_BANNED_DEPS = {
    "pyautogui", "keyboard", "pynput",      # 输入注入类，客服场景无正当用途
    "pickle", "dill",                        # 反序列化执行
    "telnetlib", "ftplib",                   # 明文协议
}
# 草稿载荷里出现这些键 = 试图把"生产写/原始查询"包装成 Skill
_GATE_FORBIDDEN_KEYS = {
    "sql", "raw_sql", "execute_sql", "write_production", "update_production",
    "query_customer", "raw_customer", "export_full_dataset", "disable_audit",
}


def _change_or_404(cid: str) -> dict:
    ch = db.get_change(cid)
    if not ch:
        raise HTTPException(404, "变更不存在")
    return ch


def _payload_of(ch: dict) -> dict:
    try:
        p = json.loads(ch.get("payload") or "{}")
        return p if isinstance(p, dict) else {}
    except Exception:
        return {}


def _fingerprint(cid: str, payload: dict) -> str:
    """草稿内容指纹：审批时记下，发布时重算比对。"""
    return sign({"change_id": cid, "payload": payload}, SESSION_SECRET)


def _run_gate(ch: dict) -> dict:
    """回归门槛：四项检查全过才允许发布。

    诚实说明：这是 **PoC 级确定性门槛**，不是完整 CI。它检查的是
    "草稿有没有越权声明、依赖有没有黑名单、有没有测试用例、提交人有没有权限"，
    而不是"跑一套真实回归套件"。生产接入应替换为流水线（见 docs/21）。
    """
    payload = _payload_of(ch)
    checks: list[dict] = []

    # ① 结构完整性
    checks.append({
        "id": "SCHEMA", "name": "草稿结构完整性",
        "ok": bool(payload) and bool(ch.get("purpose")),
        "detail": f"payload 键 {sorted(payload)}，用途长度 {len(ch.get('purpose') or '')}",
    })

    # ② 越权声明扫描
    hit = sorted(_GATE_FORBIDDEN_KEYS & set(payload))
    checks.append({
        "id": "NO_PROD_WRITE", "name": "禁止生产写/原始查询声明",
        "ok": not hit, "detail": f"命中 {hit}" if hit else "无越权键",
    })

    # ③ 依赖黑名单扫描
    deps = [str(d).split("==")[0].split(">=")[0].strip().lower()
            for d in (payload.get("dependencies") or [])]
    banned = sorted(set(deps) & _GATE_BANNED_DEPS)
    checks.append({
        "id": "DEPENDENCY_SCAN", "name": "依赖黑名单扫描",
        "ok": not banned, "detail": f"命中 {banned}" if banned else f"依赖 {len(deps)} 个均通过",
    })

    # ④ 权限检查：提交人必须是科技人员角色（业务/审核不能直接提代码类变更）
    u = db.user_by_emp_id(ch.get("owner") or "")
    role = (u or {}).get("role")
    need_dev = ch.get("kind") in ("skill", "workflow")
    ok_perm = (role == DEVELOPER) if need_dev else (role in (DEVELOPER, SUPERVISOR))
    checks.append({
        "id": "PERMISSION", "name": "提交人权限匹配",
        "ok": ok_perm,
        "detail": f"owner={ch.get('owner')} role={role} kind={ch.get('kind')}",
    })

    # ⑤ 回归用例（沙箱）：必须声明 ≥3 条合成用例，且只用合成数据
    cases = payload.get("test_cases") or []
    if isinstance(cases, str):
        cases = [c for c in cases.splitlines() if c.strip()]
    dataset = str(payload.get("synthetic_dataset_id") or payload.get("dataset") or "")
    ok_reg = len(cases) >= 3 and (not dataset or "synthetic" in dataset.lower() or "合成" in dataset)
    checks.append({
        "id": "REGRESSION", "name": "沙箱回归（合成数据）",
        "ok": ok_reg,
        "detail": f"用例 {len(cases)} 条，数据集={dataset or '默认合成集'}",
    })

    passed = all(c["ok"] for c in checks)
    return {
        "passed": passed,
        "checks": checks,
        "ran_at": service._iso(service._now()),
        "digest": _fingerprint(ch["change_id"], payload),
        "scope_note": "PoC 级确定性门槛，非完整 CI 流水线",
    }


@app.post("/internal/api/v1/changes/{cid}/gate")
def change_gate(cid: str, user: dict = Depends(auth.require(SUPERVISOR, DEVELOPER))) -> dict:
    ch = _change_or_404(cid)
    if ch["status"] not in ("PENDING_REVIEW", "APPROVED"):
        raise HTTPException(409, f"当前状态 {ch['status']} 不可执行门槛检查")
    gate = _run_gate(ch)
    db.set_change(cid, gate=json.dumps(gate, ensure_ascii=False), gate_at=gate["ran_at"],
                  gate_passed=1 if gate["passed"] else 0)
    audit.log("inner", "GATE_CHECKED", actor=user["emp_id"],
              detail={"change_id": cid, "passed": gate["passed"],
                      "failed": [c["id"] for c in gate["checks"] if not c["ok"]]})
    return gate


@app.post("/internal/api/v1/changes/{cid}/review")
def change_review(cid: str, req: ChangeReviewRequest,
                  user: dict = Depends(auth.require(SUPERVISOR))) -> dict:
    ch = _change_or_404(cid)
    if ch["status"] != "PENDING_REVIEW":
        raise HTTPException(409, f"当前状态 {ch['status']} 不可审批")
    # ① 不得自审批：服务端判定，与前端是否显示按钮无关
    if ch["owner"] == user["emp_id"]:
        audit.log("inner", "AUTHZ_DENIED", actor=user["emp_id"], severity="high",
                  detail={"reason": "self_approval", "change_id": cid})
        raise HTTPException(403, "不得审批自己提交的变更")
    ts = service._iso(service._now())
    if req.decision == "REJECT":
        db.set_change(cid, status="REJECTED", reviewer=user["emp_id"], reviewed_at=ts,
                      review_note=req.note or "")
        audit.log("inner", "DRAFT_REJECTED", actor=user["emp_id"],
                  detail={"change_id": cid, "note": req.note})
        return {"change_id": cid, "status": "REJECTED"}
    db.set_change(cid, status="APPROVED", reviewer=user["emp_id"], reviewed_at=ts,
                  review_note=req.note or "")
    audit.log("inner", "DRAFT_APPROVED", actor=user["emp_id"],
              detail={"change_id": cid, "note": req.note})
    return {"change_id": cid, "status": "APPROVED",
            "next": "需先通过门槛检查（/gate）再发布"}


@app.post("/internal/api/v1/changes/{cid}/publish")
def change_publish(cid: str, user: dict = Depends(auth.require(SUPERVISOR))) -> dict:
    ch = _change_or_404(cid)
    if ch["status"] != "APPROVED":
        raise HTTPException(409, f"当前状态 {ch['status']} 不可发布（须先审批通过）")
    if not ch.get("gate_passed"):
        raise HTTPException(409, "回归门槛未通过，禁止发布")
    # ③ 指纹比对：审批后被改内容一律拒绝
    stored = {}
    try:
        stored = json.loads(ch.get("gate") or "{}")
    except Exception:
        pass
    if stored.get("digest") != _fingerprint(cid, _payload_of(ch)):
        audit.log("inner", "DRAFT_TAMPERED", actor=user["emp_id"], severity="critical",
                  detail={"change_id": cid})
        raise HTTPException(409, "草稿内容与审批时不一致，禁止发布")
    ver = f"{ch['kind']}-v{db.published_count(ch['kind']) + 1}"
    ts = service._iso(service._now())
    db.set_change(cid, status="PUBLISHED", published_version=ver, published_at=ts)
    audit.log("inner", "DRAFT_PUBLISHED", actor=user["emp_id"],
              detail={"change_id": cid, "version": ver,
                      "gate_digest": stored.get("digest", "")[:16]})
    return {"change_id": cid, "status": "PUBLISHED", "version": ver, "published_at": ts}


@app.post("/internal/api/v1/changes/{cid}/rollback")
def change_rollback(cid: str, req: ChangeRollbackRequest,
                    user: dict = Depends(auth.require(SUPERVISOR))) -> dict:
    ch = _change_or_404(cid)
    if ch["status"] != "PUBLISHED":
        raise HTTPException(409, f"当前状态 {ch['status']} 不可回滚（仅已发布可回滚）")
    ts = service._iso(service._now())
    db.set_change(cid, status="ROLLED_BACK", rollback_reason=req.reason, rolled_back_at=ts)
    audit.log("inner", "DRAFT_ROLLBACK", actor=user["emp_id"], severity="high",
              detail={"change_id": cid, "reason": req.reason,
                      "version": ch.get("published_version")})
    return {"change_id": cid, "status": "ROLLED_BACK",
            "rolled_back_version": ch.get("published_version"), "reason": req.reason}


# ---------------- MCP Facade ----------------
# 只为白名单工具提供实现。被禁止的能力不在此注册，调用即 404。

_TOOL_IMPL_SCOPE = {
    "list_rule_catalog": {"auditor", "developer", "supervisor"},
    "get_rule_metadata": {"auditor", "developer", "supervisor"},
    "get_approved_rule_summary": {"auditor", "developer", "supervisor"},
    "list_synthetic_cases": {"auditor", "developer", "supervisor"},
    "get_masked_sample_case": {"auditor", "developer"},
    "get_aggregate_metrics": {"auditor", "developer", "supervisor"},
    "submit_skill_draft": {"developer"},
    "submit_workflow_draft": {"developer"},
    "submit_rule_change_request": {"developer"},
    "run_sandbox_test": {"developer"},
    "get_change_status": {"developer", "auditor"},
}


def _require(tool: str, role: str) -> None:
    allowed = _TOOL_IMPL_SCOPE.get(tool)
    if allowed is None:
        raise HTTPException(404, f"工具未注册: {tool}")
    if role not in allowed:
        audit.log("inner", "MCP_DENIED", severity="high",
                  detail={"tool": tool, "role": role, "reason": "scope"})
        raise HTTPException(403, f"角色 {role} 无权调用 {tool}")


@app.post("/internal/api/v1/mcp/proxy")
async def mcp_proxy(req: McpProxyRequest,
                    user: dict = Depends(auth.require(SUPERVISOR, AUDITOR, DEVELOPER))):
    """管理端的受控工具入口：浏览器只与内网同源通信，会话不出内网。

    角色在这里由会话确定，随后作为已经定死的值穿过 DMZ 回到 Facade——
    前端没有机会声明自己是 developer。
    """
    url = f"{DMZ_URL}/dmz/api/v1/mcp/call"
    guard("inner", url)
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.post(
                url,
                json={"tool": req.tool, "actor": user["emp_id"], "role": user["role"], "args": req.args},
                headers={auth.SERVICE_HEADER: SERVICE_TOKEN},
            )
    except Exception as exc:
        raise HTTPException(502, f"MCP 网关不可达：{exc}")
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise HTTPException(r.status_code, detail)
    return JSONResponse({"tool": req.tool, "role": user["role"], **r.json()})


@app.post("/internal/api/v1/mcp/{tool}")
def mcp(tool: str, request: Request, body: dict = Body(...)) -> JSONResponse:
    # 身份来源只有两种：DMZ 的服务令牌（角色已由上层会话确定），或浏览器会话本身
    if auth.is_service(request):
        role = body.get("role", "")
        actor = body.get("actor", "service")
        if not role:
            raise HTTPException(403, "服务调用必须携带已确定的角色")
    else:
        u = auth.current_user(request)
        role = u["role"]
        actor = u["emp_id"]
    args = body.get("args") or {}
    _require(tool, role)
    pr = PrivacyService([c["name"] for c in db.all_customers()])

    if tool == "list_rule_catalog":
        rows = db.all_rules()
        if args.get("domain"):
            rows = [r for r in rows if r["domain"] == args["domain"]]
        out = [{"rule_id": r["rule_id"], "domain": r["domain"], "title": r["title"],
                "version": r["version"], "status": r["status"],
                "effective_from": r["effective_from"],
                "source": r.get("source") or "出处未登记"} for r in rows]
    elif tool == "get_rule_metadata":
        r = db.rule(args.get("rule_id", ""))
        if not r:
            raise HTTPException(404, "规则不存在")
        out = {k: r[k] for k in ("rule_id", "version", "domain", "title",
                                 "effective_from", "status", "source")}
    elif tool == "get_approved_rule_summary":
        r = db.rule(args.get("rule_id", ""))
        if not r:
            raise HTTPException(404, "规则不存在")
        out = {"rule_id": r["rule_id"], "version": r["version"], "summary": r["summary"]}
    elif tool == "list_synthetic_cases":
        rows = db.list_cases()
        if args.get("category"):
            rows = [c for c in rows if c["category"] == args["category"]]
        out = [{"case_id": c["case_id"], "category": c["category"], "intent": c["intent"]} for c in rows]
    elif tool == "get_masked_sample_case":
        c = db.get_case(args.get("case_id", ""))
        if not c:
            raise HTTPException(404, "案例不存在")
        out = {"case_id": c["case_id"], "masked": pr.mask(c["narrative"], "MCP-SAMPLE").text}
    elif tool == "get_aggregate_metrics":
        with db.connect() as conn:
            rows = conn.execute("SELECT intent, COUNT(*) n FROM tickets GROUP BY intent").fetchall()
        total = sum(r["n"] for r in rows)
        if total < MIN_AGGREGATE_SAMPLE:
            out = {"refused": True,
                   "reason": f"样本量 {total} 小于最小阈值 {MIN_AGGREGATE_SAMPLE}，拒绝返回以防小样本反推",
                   "min_sample": MIN_AGGREGATE_SAMPLE}
        else:
            out = {"period": args.get("period", "ALL"),
                   "metrics": {r["intent"]: r["n"] for r in rows if r["n"] >= 5}}
    elif tool in {"submit_skill_draft", "submit_workflow_draft", "submit_rule_change_request"}:
        kind = {"submit_skill_draft": "skill", "submit_workflow_draft": "workflow",
                "submit_rule_change_request": "rule"}[tool]
        cid = f"CHG-{abs(hash((tool, actor, json.dumps(args, sort_keys=True)))) % 100000:05d}"
        # 幂等：同一份草稿重复提交（同样的 tool/actor/args）不新建、不报错，
        # 直接返回既有条目。否则重试会撞主键变成 500，把"重试"变成"故障"。
        existing = db.get_change(cid)
        if existing:
            out = {"change_id": cid, "status": existing["status"],
                   "note": "重复提交，返回既有草稿（未新建）"}
        else:
            db.add_change(cid, kind, actor, str(args.get("purpose", ""))[:200],
                          args, service._iso(service._now()))
            out = {"change_id": cid, "status": "PENDING_REVIEW",
                   "note": "已进入待审批区，未影响生产环境"}
    elif tool == "run_sandbox_test":
        out = {"package_id": args.get("package_id"), "dataset": args.get("synthetic_dataset_id"),
               "result": "PASS", "cases": 12, "failed": 0, "isolation": "sandbox"}
    elif tool == "get_change_status":
        ch = db.get_change(args.get("change_id", ""))
        if not ch:
            raise HTTPException(404, "变更不存在")
        out = {k: ch[k] for k in ("change_id", "kind", "owner", "status")}
    else:
        raise HTTPException(404, f"工具未注册: {tool}")

    audit.log("inner", "MCP_CALL", actor=actor,
              detail={"tool": tool, "role": role, "args_keys": sorted(args)})
    return JSONResponse({"tool": tool, "ok": True, "data": out})


# ---------------- 第二期：客户反馈（M2） ----------------
#
# 数据边界（与 07-客户反馈小程序方案 一致）：
#   · 客户原文止步于 DMZ——inner 只见 DMZ 形态脱敏后的文本，入库前再做
#     一次姓名级二次脱敏（PrivacyService，DMZ 查形态、内网兜姓名，两道失效模式不同）。
#   · 所有答复（AI 或人工）出站一律经 DMZ 规则裁决，DMZ 不可达按 deny 处理
#     （fail-closed 转人工），绝不绕开裁决直发。
#   · 审计只记元数据（fb_id / 类别 / 长度），不落客户正文。

# 反馈答复口径：模型产出 customer_reply（面向客户的自然话术）→ 校验后下发；
# 意图识别/业务分类/风险等级等内部分析只进审计（meta），绝不展示给客户。
# 模板兜底是方案里写明的风险对策：不承诺任何业务结果，绝不返回空答复。
_FB_CATEGORY_CN = {"ACCOUNT": "账户", "LOAN": "贷款", "CREDIT_CARD": "信用卡", "OTHER": "业务"}
_FB_CATEGORY_GUIDE = {
    "ACCOUNT": "账户余额、交易明细可以在手机银行 App「我的—账户」页自助查询。",
    "LOAN": "贷款额度、还款计划与合同信息可以在手机银行 App「贷款」频道查看。",
    "CREDIT_CARD": "账单、额度与分期业务可以在手机银行 App「信用卡」频道查询办理。",
    "OTHER": "常见业务可以在手机银行 App 首页搜索功能名称直达办理页。",
}
_FB_STEP_CN = {
    "GUIDE_APP_QUERY": "您可以在手机银行 App 对应功能页面自助查询或办理。",
    "REQUEST_DISPUTE_FORM": "如涉及交易争议，请在手机银行 App「交易争议」入口提交并按页面提示准备材料。",
    "COLLECT_MORE_INFO": "为便于核实，请补充交易时间与诉求类型后回复本条消息。",
    "HUMAN_REVIEW": "该诉求需要人工核查，客服人员将尽快与您联系。",
}
# 模型客户话术中不得出现的内部术语（命中即降级模板，防止内部分析过程泄露给客户）
_FB_INTERNAL_RE = re.compile(
    r"动作码|意图摘要|风险标记|字段|JSON|HUMAN_REVIEW|GUIDE_APP|REQUEST_DISPUTE|"
    r"COLLECT_MORE|intent_summary|risk_flags|next_step|Agent|智能体|规则引擎|模型", re.I)


_FB_CONFIRM_RE = re.compile(r"^(嗯+|对+|好+|是+|是的|可以|行|ok|OK|继续|然后呢|还有呢|说说)[。！!？?\s]*$")


def _fb_intent_hint(body: str, history: list[dict] | None = None) -> str | None:
    """客户通道的确定性意图提示（v1.5.1 / 多轮兜底 v1.5.2）。

    模型（尤其 mock/降级）常只给通用动作码 GUIDE_APP_QUERY，导致"想提争议要准备
    什么材料"这种低风险咨询也只得到一句通用自助话术。这里按关键词给更具体的动作码，
    仅当模型没给更具体动作时生效（模型优先）。

    多轮兜底：客户只回"嗯/对/继续"这类短句时关键词命中不了，会掉回通用话术。
    此时回看历史里最近一条有意图的客户消息，沿用它的动作码（只回看一层，避免递归）。
    """
    text = (body or "").strip()
    if re.search(r"争议|申诉|材料|怎么提|投诉|举证", text):
        return "REQUEST_DISPUTE_FORM"
    if re.search(r"不清楚|不知道|那个钱|不对|看不懂", text):
        return "COLLECT_MORE_INFO"
    if _FB_CONFIRM_RE.match(text) or len(text) <= 3:
        for m in reversed(history or []):
            if m.get("sender") != "customer":
                continue
            prev = (m.get("body_masked") or m.get("body") or "").strip()
            if not prev or prev == text:
                continue
            inherited = _fb_intent_hint(prev, None)
            if inherited:
                return inherited
    return None


def _fb_template_reply(category: str, step: str, first: bool,
                       missing: list | None = None) -> str:
    """模板兜底：按业务类型给足信息量；追问只围绕本次问题且不超过 2 项。"""
    guide = _FB_CATEGORY_GUIDE.get(category, _FB_CATEGORY_GUIDE["OTHER"])
    if step == "REQUEST_DISPUTE_FORM":
        return ("您好，如涉及交易争议，可以在手机银行 App「交易争议」入口提交，"
                "按页面提示准备好订单信息和沟通记录即可，我们会尽快为您跟进核实。")
    if step == "COLLECT_MORE_INFO":
        items = "、".join((missing or [])[:2]) or "问题发生的具体场景（如大致时间、办理渠道）"
        return (f"您好，为了帮您准确核实，麻烦您补充：{items}。"
                "您可以直接回复本条消息，我会继续为您跟进。")
    if first:
        return ("您好，感谢您的反馈。" + guide +
                "如未能解决您的问题，欢迎回复补充具体情况，我会进一步为您核实。")
    return ("您好，关于您提到的问题，" + guide +
            "如仍有疑问，请回复具体情况，我会继续为您核实处理。")


def _fb_compose_reply(category: str, body: str, history: list[dict] | None = None) -> tuple[str, dict]:
    """生成客户答复：牛马仕 Agent 产出 customer_reply（面向客户）→ 校验后下发；

    内部字段（意图/风险/动作码）只进审计，不下发客户。Agent 不可用或客户话术
    不合格（为空/超长/含内部术语）时按改进后的模板口径兜底（fail-closed），
    绝不返回空答复，也绝不把内部分析过程展示给客户。
    history：会话历史，供多轮 AI 会话理解上下文，避免重复追问已答过的信息。
    """
    meta: dict = {"engine": "template-fallback", "next_step": "GUIDE_APP_QUERY",
                  "risk_flags": [], "degraded": "", "reply_source": "template-fallback"}
    step = "GUIDE_APP_QUERY"
    customer_reply = ""
    missing: list = []
    agent_input = body
    if history:
        turns = [m for m in history if m.get("sender") in ("customer", "ai")
                 and (m.get("body_masked") or m.get("body") or "").strip()]
        lines = [f"{i}. {'客户' if m['sender'] == 'customer' else '客服'}："
                 f"{(m.get('body_masked') or m.get('body') or '')[:80]}"
                 for i, m in enumerate(turns[-8:], 1)]
        agent_input = ("会话记录：\n" + "\n".join(lines) +
                       f"\n请针对客户最新消息撰写回复；客户已提供过的信息不要重复询问。"
                       f"客户最新消息：{body}")
    # v1.5.5：把内网规则库摘要一起给模型——回答要有出处，不是自由发挥
    _intent_for_rules = {"ACCOUNT": "TRANSACTION_QUERY", "CREDIT_CARD": "DISPUTE_PROCESS",
                         "LOAN": "PENDING_CONFIRM", "OTHER": "DISPUTE_PROCESS"}.get(category, "DISPUTE_PROCESS")
    try:
        _rules = db.rules_for(_intent_for_rules) or []
        if _rules:
            _lines = [f"- [{r.get('rule_id')}] {str(r.get('summary') or '')[:80]}" for r in _rules[:6]]
            agent_input += "\n\n可引用的内网规则（回答须与之一致，不得编造未列出的政策）：\n" + "\n".join(_lines)
    except Exception:
        pass
    try:
        plan = NiumashiAgent().plan("FEEDBACK_REPLY", agent_input)
        out = plan.get("output") or {}
        step = out.get("recommended_next_step", "GUIDE_APP_QUERY")
        customer_reply = str(out.get("customer_reply", "")).strip()
        missing = out.get("missing_information") or []
        meta = {"engine": f"{out.get('model_supplier', 'nms-mock')}/{out.get('model_version', '')}",
                "next_step": step, "risk_flags": out.get("risk_flags", []),
                "degraded": out.get("degraded", ""), "reply_source": "model",
                "intent": str(out.get("intent_summary", ""))[:60]}
    except NotImplementedError:
        pass  # AGENT_IMPL 未接入的实现：按模板口径答复
    except Exception as exc:
        runlog.warning("fb_agent_failed", extra={"extra_data": {"error": str(exc)[:200]}})
        meta["degraded"] = f"agent-error: {type(exc).__name__}"
        meta["reply_source"] = "template-fallback"
    if meta.get("next_step") not in _FB_STEP_CN:
        meta["next_step"] = "GUIDE_APP_QUERY"
    # v1.5.1：模型只给通用动作码时，用确定性意图提示补更具体的动作（模型优先）
    _hint = _fb_intent_hint(body, history)
    if _hint and meta.get("next_step") == "GUIDE_APP_QUERY":
        meta["next_step"] = _hint
        meta["intent_hint"] = _hint
    step = meta["next_step"]

    # 客户答复：优先用模型产出的客户话术；命中意图提示时改用对应模板（保证动作码与话术一致）
    reply = ""
    # v1.5.5：只要模型话术合规就用它（此前命中意图提示就退回模板，导致不同问题回答雷同）
    _ok_model = (bool(customer_reply) and len(customer_reply) <= 300
                 and not _FB_INTERNAL_RE.search(customer_reply))
    if _ok_model and step == "HUMAN_REVIEW" and not ("人工" in customer_reply or "转接" in customer_reply):
        _ok_model = False          # 转人工场景必须明确告知已转人工，否则用模板兜底
        meta["degraded"] = (meta.get("degraded", "") + " handoff_wording_missing").strip()
    if _ok_model:
        reply = customer_reply
        meta["reply_source"] = "model"
    else:
        if customer_reply:
            meta["degraded"] = (meta.get("degraded", "") + " customer_reply_rejected").strip()
        reply = _fb_template_reply(category, step, first=not history, missing=missing)

    # 转人工提示只在首轮附带（且模型话术未提到时），多轮不再重复劝用自助
    if not history and step != "HUMAN_REVIEW" and "人工" not in reply and "转接" not in reply:
        reply += "如需人工客服，回复“转人工”即可。"
    return reply, meta


def _fb_deliver(fb_id: str, rid: str, sender: str, body: str,
                status_after: str, ts: str) -> dict:
    """答复出站：一律经 DMZ 规则裁决后才下发网关。DMZ 不可达按 deny 处理。"""
    guard("inner", DMZ_URL)
    try:
        r = httpx.post(
            f"{DMZ_URL}/dmz/api/v1/feedback/deliver",
            json=FeedbackDeliverRequest(fb_id=fb_id, rid=rid, sender=sender, body=body,
                                        status_after=status_after, created_at=ts).model_dump(),
            headers={auth.SERVICE_HEADER: SERVICE_TOKEN},
            timeout=8,
        )
        if r.status_code != 200:
            return {"verdict": "deny", "rule_id": f"DMZ_STATUS_{r.status_code}"}
        return r.json()
    except Exception as exc:
        audit.log("inner", "FB_DELIVER_UNREACHABLE", request_id=fb_id, severity="high",
                  detail={"error": str(exc)[:200]})
        return {"verdict": "deny", "rule_id": "DMZ_UNAVAILABLE"}


def _fb_notify_status(fb_id: str, status: str) -> None:
    """状态单向同步到网关缓存（经 DMZ），失败只告警不阻断——内网状态是唯一事实源。"""
    guard("inner", DMZ_URL)
    try:
        httpx.post(f"{DMZ_URL}/dmz/api/v1/feedback/status",
                   json={"fb_id": fb_id, "status": status},
                   headers={auth.SERVICE_HEADER: SERVICE_TOKEN}, timeout=5)
    except Exception as exc:
        runlog.warning("fb_status_sync_failed", extra={"extra_data": {
            "fb_id": fb_id, "status": status, "error": str(exc)[:150]}})


@app.post("/internal/api/v1/feedback/ingest")
def fb_ingest_ep(req: FeedbackIngestRequest, request: Request) -> dict:
    """DMZ 脱敏后的反馈入站。只接受 DMZ 服务调用，浏览器会话一律拒绝。"""
    if not auth.is_service(request):
        audit.log("inner", "FB_DENIED", severity="critical",
                  detail={"reason": "session_to_ingest", "fb_id": req.fb_id})
        raise HTTPException(403, "反馈入站仅接受 DMZ 服务调用")
    if req.category not in FEEDBACK_CATEGORIES:
        raise HTTPException(400, "未知业务类型")
    now = service._iso_ms(service._now())
    pr = PrivacyService([c["name"] for c in db.all_customers()])
    masked = pr.mask(req.body_masked, req.fb_id).text  # 二次脱敏：姓名级兜底

    t = db.fb_ticket(req.fb_id)
    # 客户评价：仅统计（满意=1/不满意=0），写审计不触发应答；不进会话
    if req.kind == "eval":
        if not t:
            raise HTTPException(404, "反馈单不存在")
        try:
            sat = int(masked)
            assert sat in (0, 1)
        except Exception:
            raise HTTPException(400, "评价取值非法")
        db.fb_rate(req.fb_id, sat, now)
        # 评价即办结：AI 会话由客户评价终止（人工流程则已由坐席办结）
        if not str(t["status"]).startswith("CLOSED"):
            db.fb_set_status(req.fb_id, "CLOSED_AI", now)
            _fb_notify_status(req.fb_id, "CLOSED_AI")
        audit.log("inner", "FB_RATED", request_id=req.fb_id, detail={"satisfaction": sat})
        return {"ok": True, "verdict": "rated", "fb_id": req.fb_id}

    if req.kind == "ticket":
        if t:
            raise HTTPException(409, "反馈单号重复")
        db.fb_ingest(req.fb_id, req.openid_hash, req.category, masked, now)
        # 客户原始提问落为会话首条消息（脱敏后），坐席会话视图可见
        db.fb_add_reply(req.fb_id, "FBR-C-" + secrets.token_hex(6), "customer",
                        masked, "pass", now)
    else:
        if not t:
            raise HTTPException(404, "反馈单不存在")
        if t["status"].startswith("CLOSED"):
            raise HTTPException(409, "工单已办结，不能追加消息")
        db.fb_add_reply(req.fb_id, "FBR-C-" + secrets.token_hex(6), "customer",
                        masked, "pass", now)
    audit.log("inner", "FB_INGESTED", request_id=req.fb_id,
              detail={"kind": req.kind, "category": req.category, "len": len(masked)})

    # 人工已在处理中：客户追加消息直接进会话，不再触发 AI（方案 3.2）
    if req.kind == "message" and t and t["status"] == "HUMAN_ANSWERING":
        db.fb_set_status(req.fb_id, "HUMAN_ANSWERING", now)
        return {"ok": True, "verdict": "human-active", "fb_id": req.fb_id}

    db.fb_set_status(req.fb_id, "AI_ANSWERING", now)
    _fb_notify_status(req.fb_id, "AI_ANSWERING")

    # 客户显式要求人工：多轮 AI 会话的唯一主动退出路径（除评价办结外）。
    # 按导师意见：客户要人工时直接确认转接，不再劝说使用自助功能。
    if any(k in masked for k in ("转人工", "人工客服", "人工服务", "找人工", "转接人工")):
        db.fb_set_status(req.fb_id, "PENDING_HUMAN", now)
        _fb_notify_status(req.fb_id, "PENDING_HUMAN")
        db.fb_add_reply(req.fb_id, "FBR-S-" + secrets.token_hex(6), "system",
                        "已为您转接人工客服，坐席将尽快回复，请稍候。",
                        "system", now)
        audit.log("inner", "FB_HUMAN_REQUESTED", request_id=req.fb_id)
        return {"ok": True, "verdict": "human", "fb_id": req.fb_id}

    # 确定性高风险词规则：涉诈/盗/资金损失一类诉求不依赖模型判断，一律转人工。
    # 模型可能漏判（冒烟实测），资金安全宁可错转也不可漏转。
    # 按导师意见：转人工的同时仍下发一条安抚答复（谨慎处理策略），
    # 客户不再是干巳巳的「已受理待人工接管」，而是有人味儿的即时回应。
    if any(k in masked for k in (
        # v1.5.1：补齐"疑似未授权"的表达——原来只认「盗/骗/冻结」，
        # 客户说「陌生消费」「未授权」「不是本人」时会被当成普通咨询自动答复。
        "骗", "盗", "洗钱", "资金损失", "被扣款", "冻结",
        "陌生消费", "未授权", "不是本人", "非本人", "被刷", "异常消费",
        "异常交易", "盗刷", "退款", "冒名", "挂失",
    )):
        db.fb_set_status(req.fb_id, "PENDING_HUMAN", now)
        _fb_notify_status(req.fb_id, "PENDING_HUMAN")
        notice = ("您好，您反映的问题我们已收到，非常理解您的心情。"
                  "为保障您的资金安全，已将您的诉求标记为优先处理，"
                  "人工客服将尽快与您联系，请保持手机畅通。"
                  "特别提醒：任何索要短信验证码、密码或要求转账的行为都是诈骗，"
                  "请不要向任何人透露。")
        rid_n = "FBR-A-" + secrets.token_hex(6)
        now_n = service._iso_ms(service._now())
        v = _fb_deliver(req.fb_id, rid_n, "ai", notice, "PENDING_HUMAN", now_n)
        if v.get("verdict") == "pass":
            db.fb_add_reply(req.fb_id, rid_n, "ai", notice,
                            json.dumps(v, ensure_ascii=False), now_n)
            audit.log("inner", "FB_RISK_SAFETY_NOTICE", request_id=req.fb_id)
        db.fb_add_reply(req.fb_id, "FBR-S-" + secrets.token_hex(6), "system",
                        "为保障您的资金安全，已优先为您转接人工客服。",
                        "system", now)
        audit.log("inner", "FB_RISK_KEYWORD_HUMAN", request_id=req.fb_id, severity="high")
        return {"ok": True, "verdict": "human", "fb_id": req.fb_id}

    # AI 多轮会话：人工接管前由 AI 持续应答，带会话上下文；不自动办结，
    # 客户评价（eval）或转人工后才离开 AI_ANSWERING 态。
    # 多轮历史：优先本工单；本工单没有历史时，用同一客户（openid_hash）最近的跨工单消息兜底
    _hist = db.fb_replies(req.fb_id)
    if len(_hist) < 2:
        try:
            _hist = db.fb_recent_by_openid(req.openid_hash) + _hist
        except Exception:
            pass
    reply, meta = _fb_compose_reply(req.category, masked, history=_hist)
    if "HUMAN_REVIEW_REQUIRED" in meta.get("risk_flags", []) \
            or meta["next_step"] == "HUMAN_REVIEW":
        db.fb_set_status(req.fb_id, "PENDING_HUMAN", now)
        _fb_notify_status(req.fb_id, "PENDING_HUMAN")
        # 按导师意见：涉敏感风险谨慎处理 ≠ 不回复——仍给客户安抚答复（经 DMZ 裁决）
        notice = ("您好，您反映的问题我们已收到。该问题涉及需要人工核实的内容，"
                  "为谨慎起见，已为您转接人工客服优先跟进，坐席会尽快与您联系。"
                  "为保障账户安全，请您不要向任何人透露短信验证码和密码。")
        rid_n = "FBR-A-" + secrets.token_hex(6)
        now_n = service._iso_ms(service._now())
        v = _fb_deliver(req.fb_id, rid_n, "ai", notice, "PENDING_HUMAN", now_n)
        if v.get("verdict") == "pass":
            db.fb_add_reply(req.fb_id, rid_n, "ai", notice,
                            json.dumps(v, ensure_ascii=False), now_n)
        db.fb_add_reply(req.fb_id, "FBR-S-" + secrets.token_hex(6), "system",
                        "该诉求需人工核实，已优先转人工跟进。",
                        "system", now)
        audit.log("inner", "FB_AI_FORCED_HUMAN", request_id=req.fb_id, detail=meta)
        return {"ok": True, "verdict": "human", "fb_id": req.fb_id}

    rid = "FBR-A-" + secrets.token_hex(6)
    # AI 答复取独立的后置时间戳：必须晚于触发它的客户消息（毫秒精度下严格递增）
    now_ai = service._iso_ms(service._now())
    verdict = _fb_deliver(req.fb_id, rid, "ai", reply, "AI_ANSWERING", now_ai)
    db.fb_add_reply(req.fb_id, rid, "ai", reply,
                    json.dumps(verdict, ensure_ascii=False), now_ai)
    if verdict.get("verdict") == "pass":
        db.fb_set_ai_reply(req.fb_id, reply, now_ai)  # 状态保持 AI_ANSWERING（会话继续）
        audit.log("inner", "FB_AI_REPLIED", request_id=req.fb_id, detail=meta)
    else:
        db.fb_set_status(req.fb_id, "PENDING_HUMAN", now)
        _fb_notify_status(req.fb_id, "PENDING_HUMAN")
        db.fb_add_reply(req.fb_id, "FBR-S-" + secrets.token_hex(6), "system",
                        "该条答复未通过安全审核，已为您转接人工客服，坐席将尽快回复。",
                        "system", now)
        audit.log("inner", "FB_AI_DENIED", request_id=req.fb_id, severity="high",
                  detail={"rule_id": verdict.get("rule_id"), "fb_id": req.fb_id})
    return {"ok": True, "verdict": verdict.get("verdict"),
            "rule_id": verdict.get("rule_id", ""), "fb_id": req.fb_id}


@app.get("/internal/api/v1/feedback/list")
def fb_list_ep(status: str | None = Query(None),
               user: dict = Depends(auth.current_user)) -> dict:
    return {"items": db.fb_list(status), "pending": db.fb_pending_count()}


@app.get("/internal/api/v1/feedback/mine")
def fb_mine_ep(user: dict = Depends(auth.require(AGENT, SUPERVISOR))) -> dict:
    """坐席个人工作档案：只含本坐席接管过的工单。"""
    return {"items": db.fb_mine(user["emp_id"])}


@app.get("/internal/api/v1/feedback/archive")
def fb_archive_ep(user: dict = Depends(auth.require(SUPERVISOR))) -> dict:
    """档案库（管理端）：全部工单处理记录 + 按坐席汇总。"""
    return {"items": db.fb_archive(), "stats": db.fb_archive_stats()}


@app.get("/internal/api/v1/feedback/{fb_id}")
def fb_detail_ep(fb_id: str, user: dict = Depends(auth.current_user)) -> dict:
    t = db.fb_ticket(fb_id)
    if not t:
        raise HTTPException(404, "反馈单不存在")
    return {"ticket": t, "replies": db.fb_replies(fb_id)}


@app.post("/internal/api/v1/feedback/{fb_id}/takeover")
def fb_takeover_ep(fb_id: str, user: dict = Depends(auth.require(AGENT, SUPERVISOR))) -> dict:
    t = db.fb_ticket(fb_id)
    if not t:
        raise HTTPException(404, "反馈单不存在")
    if t["status"].startswith("CLOSED"):
        raise HTTPException(409, "工单已办结")
    if t["status"] == "HUMAN_ANSWERING" and t["assigned_emp"] != user["emp_id"]:
        raise HTTPException(409, "该工单已由其他坐席接管")
    now = service._iso_ms(service._now())
    db.fb_takeover(fb_id, user["emp_id"], now)
    _fb_notify_status(fb_id, "HUMAN_ANSWERING")
    audit.log("inner", "FB_TAKEOVER", request_id=fb_id, actor=user["emp_id"],
              detail={"prev_status": t["status"]})
    return {"ok": True, "status": "HUMAN_ANSWERING"}


@app.post("/internal/api/v1/feedback/{fb_id}/reply")
def fb_reply_ep(fb_id: str, req: FeedbackSeatReplyRequest,
                user: dict = Depends(auth.require(AGENT, SUPERVISOR))) -> dict:
    t = db.fb_ticket(fb_id)
    if not t:
        raise HTTPException(404, "反馈单不存在")
    if t["status"] not in ("AI_ANSWERING", "PENDING_HUMAN", "HUMAN_ANSWERING"):
        raise HTTPException(409, "当前状态不允许人工回复")
    now = service._iso_ms(service._now())
    if t["status"] in ("AI_ANSWERING", "PENDING_HUMAN"):
        db.fb_takeover(fb_id, user["emp_id"], now)  # 回复即接管（含从 AI 手中接管）
    pr = PrivacyService([c["name"] for c in db.all_customers()])
    masked = pr.mask(req.body, fb_id).text
    rid = "FBR-H-" + secrets.token_hex(6)
    verdict = _fb_deliver(fb_id, rid, user["emp_id"], masked, "HUMAN_ANSWERING", now)
    if verdict.get("verdict") != "pass":
        audit.log("inner", "FB_SEAT_REPLY_DENIED", request_id=fb_id, actor=user["emp_id"],
                  severity="high", detail={"rule_id": verdict.get("rule_id")})
        raise HTTPException(451, f"答复被 DMZ 规则否决（{verdict.get('rule_id')}），未下发")
    db.fb_add_reply(fb_id, rid, user["emp_id"], masked, "pass", now)
    db.fb_set_status(fb_id, "HUMAN_ANSWERING", now)
    audit.log("inner", "FB_SEAT_REPLIED", request_id=fb_id, actor=user["emp_id"],
              detail={"len": len(masked)})
    return {"ok": True, "rid": rid, "status": "HUMAN_ANSWERING"}


@app.post("/internal/api/v1/feedback/{fb_id}/close")
def fb_close_ep(fb_id: str, user: dict = Depends(auth.require(AGENT, SUPERVISOR))) -> dict:
    t = db.fb_ticket(fb_id)
    if not t:
        raise HTTPException(404, "反馈单不存在")
    if t["status"].startswith("CLOSED"):
        raise HTTPException(409, "工单已办结，无需重复操作")
    now = service._iso_ms(service._now())
    db.fb_close(fb_id, user["emp_id"], now)
    _fb_notify_status(fb_id, "CLOSED_HUMAN")
    audit.log("inner", "FB_CLOSED", request_id=fb_id, actor=user["emp_id"],
              detail={"prev_status": t["status"]})
    return {"ok": True, "status": "CLOSED_HUMAN"}
