# app.py  (TCS Production Support Gateway - Render + Cloudflare Pages friendly)

import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from difflib import SequenceMatcher
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Optional: Tavily fallback (internet)
# If you don't want internet fallback, simply don't set TAVILY_API_KEY in Render.
try:
    from tavily import TavilyClient  # requirements: tavily-python
except Exception:
    TavilyClient = None


# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent

INCIDENTS_FILE = BASE_DIR / "incidents.json"   # ServiceNow mock dataset
MAILBOX_FILE   = BASE_DIR / "mailbox.json"     # Mailbox dataset

# Server log files are optional; if you mount logs, point LOG_ROOT to that folder.
LOG_ROOT = Path(os.environ.get("LOG_ROOT", str(BASE_DIR / "server_logs")))

# Similarity thresholds (tune later)
INC_MIN_SCORE  = float(os.environ.get("INC_MIN_SCORE", "0.62"))
MAIL_MIN_SCORE = float(os.environ.get("MAIL_MIN_SCORE", "0.58"))
LOG_MIN_SCORE  = float(os.environ.get("LOG_MIN_SCORE", "0.55"))

# Tavily fallback (internet)
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "").strip()
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if (TAVILY_API_KEY and TavilyClient) else None


# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(title="TCS Production Support Gateway", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later: restrict to Cloudflare Pages domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# MODELS
# =============================================================================
class AnalyzeReq(BaseModel):
    error_text: str = Field(..., description="Paste error / stack trace / log text")

class BlockSummary(BaseModel):
    found: bool
    title: str
    summary: str = ""
    confidence: Optional[float] = None

class ServiceNowDetails(BaseModel):
    incident_id: Optional[str] = None
    error_signature: Optional[str] = None
    root_cause: Optional[str] = None
    resolution: List[str] = []
    validation: List[str] = []

class MailDetails(BaseModel):
    thread_id: Optional[str] = None
    subject: Optional[str] = None
    from_addr: Optional[str] = None
    received_at: Optional[str] = None
    alert_body: Optional[str] = None
    resolution_body: Optional[str] = None
    related_incident_id: Optional[str] = None

class LogDetails(BaseModel):
    log_path: Optional[str] = None
    timestamp: Optional[str] = None
    file_exists: bool = False
    excerpt: Optional[str] = None

class FallbackDetails(BaseModel):
    used: bool = False
    mode: str = "none"   # "internet" | "generic"
    answer: str = ""
    sources: List[Dict[str, str]] = []

class AnalyzeResp(BaseModel):
    # Tab 1 blocks (summary)
    servicenow: BlockSummary
    mail: BlockSummary
    server_logs: BlockSummary
    fallback: BlockSummary

    # Tab 2 details (drill-down)
    details: Dict[str, Any]


# =============================================================================
# LOADERS
# =============================================================================
def _safe_load_json(path: Path, default: Any):
    if not path.exists():
        return default
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default

def load_incidents() -> List[Dict[str, Any]]:
    data = _safe_load_json(INCIDENTS_FILE, default=[])
    if isinstance(data, list):
        return data
    return []

def load_mailbox_messages() -> List[Dict[str, Any]]:
    mb = _safe_load_json(MAILBOX_FILE, default={})
    # mailbox.json format: {"folders": {"Inbox": [..], "Sent Items":[..], ...}}
    folders = (mb or {}).get("folders", {})
    msgs: List[Dict[str, Any]] = []
    if isinstance(folders, dict):
        for _, arr in folders.items():
            if isinstance(arr, list):
                msgs.extend(arr)
    return msgs


# =============================================================================
# SIMILARITY + HELPERS
# =============================================================================
def normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def similarity(a: str, b: str) -> float:
    a2, b2 = normalize(a), normalize(b)
    if not a2 or not b2:
        return 0.0
    return SequenceMatcher(None, a2, b2).ratio()

def extract_log_path(text: str) -> Optional[str]:
    if not text:
        return None
    # Common pattern in your mailbox bodies: "LogPath: /var/log/..."
    m = re.search(r"LogPath:\s*([^\s]+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Generic linux path fallback
    m2 = re.search(r"(/var/log/[A-Za-z0-9_\-./]+)", text)
    if m2:
        return m2.group(1).strip()

    return None

def read_log_excerpt(log_path: str, max_lines: int = 80) -> Tuple[bool, Optional[str]]:
    """
    Attempts to read a log file from LOG_ROOT mapped folder.
    If log_path is absolute like /var/log/..., we map it under LOG_ROOT by stripping leading slash.
    Example: /var/log/aurora/prod/x.log -> LOG_ROOT/var/log/aurora/prod/x.log
    """
    if not log_path:
        return False, None

    rel = log_path.lstrip("/")  # make relative
    candidate = LOG_ROOT / rel

    if not candidate.exists() or not candidate.is_file():
        return False, None

    try:
        lines = candidate.read_text(encoding="utf-8", errors="ignore").splitlines()
        tail = lines[-max_lines:] if len(lines) > max_lines else lines
        excerpt = "\n".join(tail)
        return True, excerpt
    except Exception:
        return True, None


# =============================================================================
# MATCHERS
# =============================================================================
def match_servicenow(error_text: str) -> Tuple[Optional[Dict[str, Any]], float]:
    incidents = load_incidents()
    best = None
    best_score = 0.0

    for inc in incidents:
        sig = inc.get("error_signature", "") or ""
        sc = similarity(error_text, sig)
        if sc > best_score:
            best_score = sc
            best = inc

    if best and best_score >= INC_MIN_SCORE:
        return best, best_score
    return None, best_score

def group_threads(messages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    threads: Dict[str, List[Dict[str, Any]]] = {}
    for m in messages:
        tid = (m.get("thread_id") or f"MSG-{m.get('id')}")  # fallback
        threads.setdefault(tid, []).append(m)
    # sort by received_at if possible
    for tid, arr in threads.items():
        arr.sort(key=lambda x: (x.get("received_at") or ""))
    return threads

def match_mail(error_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], float]:
    """
    Returns: (best_alert_msg, best_resolution_msg, score)
    We search all messages subject+body and pick best thread.
    Then within that thread choose the ALERT and RESOLUTION messages.
    """
    msgs = load_mailbox_messages()
    if not msgs:
        return None, None, 0.0

    threads = group_threads(msgs)

    best_tid = None
    best_score = 0.0

    for tid, arr in threads.items():
        # compute thread score as max over messages
        thread_best = 0.0
        for m in arr:
            text = f"{m.get('subject','')}\n{m.get('body','')}"
            sc = similarity(error_text, text)
            thread_best = max(thread_best, sc)
        if thread_best > best_score:
            best_score = thread_best
            best_tid = tid

    if not best_tid or best_score < MAIL_MIN_SCORE:
        return None, None, best_score

    arr = threads[best_tid]
    alert = None
    resolution = None
    for m in arr:
        if (m.get("type") or "").upper() == "ALERT":
            alert = m
        if (m.get("type") or "").upper() == "RESOLUTION":
            resolution = m

    # If no explicit types, still return first/last
    if not alert and arr:
        alert = arr[0]
    if not resolution and arr:
        resolution = arr[-1] if len(arr) > 1 else None

    return alert, resolution, best_score

def match_server_logs(error_text: str, mail_alert: Optional[Dict[str, Any]]) -> Tuple[Optional[LogDetails], float]:
    """
    Server log match strategy:
    - Prefer LogPath from the ALERT email body (because your mailbox has it)
    - Else try extracting path from pasted error text
    - Attempt to read excerpt if logs exist under LOG_ROOT
    """
    candidate_path = None

    if mail_alert:
        candidate_path = extract_log_path(mail_alert.get("body", ""))

    if not candidate_path:
        candidate_path = extract_log_path(error_text)

    if not candidate_path:
        return None, 0.0

    # Confidence is "moderate" because we did a direct extraction
    conf = 0.70

    exists, excerpt = read_log_excerpt(candidate_path)
    ts = None
    if mail_alert and mail_alert.get("received_at"):
        ts = str(mail_alert.get("received_at"))

    details = LogDetails(
        log_path=candidate_path,
        timestamp=ts,
        file_exists=bool(exists),
        excerpt=excerpt if excerpt else None
    )

    return details, conf


# =============================================================================
# FALLBACK
# =============================================================================
def internet_fallback(error_text: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Uses Tavily if configured. Keeps it short + source list.
    """
    if not tavily_client:
        return "", []

    q = f"Troubleshoot this production error and suggest resolution steps:\n\n{error_text}"
    result = tavily_client.search(
        query=q,
        max_results=5,
        include_answer=True,
        include_sources=True
    )

    answer = (result.get("answer") or "").strip()
    sources = []
    for r in (result.get("results") or []):
        sources.append({
            "title": (r.get("title") or "source").strip(),
            "url": (r.get("url") or "").strip()
        })
    return answer, sources

def generic_fallback(error_text: str) -> str:
    """
    If internet is disabled, provide a safe, structured “what to check” without inventing facts.
    """
    # Extract a likely exception line for readability
    first_line = (error_text.strip().splitlines()[:1] or [""])[0]
    return (
        "No internal evidence found in ServiceNow, Mail, or Server Logs.\n\n"
        "Suggested L3 triage (generic):\n"
        "1) Identify the failing component/service and the first exception line.\n"
        "2) Check recent deployments/config changes for that service.\n"
        "3) Look for correlated time window in logs/metrics (latency, errors, retries, GC, DB locks).\n"
        "4) Validate upstream dependencies (network, DNS, TLS, DB, API timeouts).\n"
        "5) If data-related: validate schema changes, nulls, duplicates, idempotency keys.\n\n"
        f"Observed error line: {first_line}\n"
    )


# =============================================================================
# ENDPOINTS
# =============================================================================
@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "TCS Production Support Gateway",
        "incidents_file_exists": INCIDENTS_FILE.exists(),
        "mailbox_file_exists": MAILBOX_FILE.exists(),
        "log_root": str(LOG_ROOT),
        "tavily_configured": bool(tavily_client),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

@app.post("/analyze", response_model=AnalyzeResp)
def analyze(req: AnalyzeReq):
    error_text = (req.error_text or "").strip()
    if not error_text:
        return AnalyzeResp(
            servicenow=BlockSummary(found=False, title="ServiceNow", summary="No input provided."),
            mail=BlockSummary(found=False, title="Mail", summary="No input provided."),
            server_logs=BlockSummary(found=False, title="Server Logs", summary="No input provided."),
            fallback=BlockSummary(found=False, title="Fallback", summary="No input provided."),
            details={}
        )

    # 1) ServiceNow
    inc, inc_score = match_servicenow(error_text)
    sn_found = inc is not None
    sn_summary = "No prior ServiceNow ticket found."
    sn_details = ServiceNowDetails()

    if sn_found:
        sn_details = ServiceNowDetails(
            incident_id=inc.get("incident_id"),
            error_signature=inc.get("error_signature"),
            root_cause=inc.get("root_cause"),
            resolution=list(inc.get("resolution") or []),
            validation=list(inc.get("validation") or []),
        )
        sn_summary = f"Match: {sn_details.incident_id} • {sn_details.root_cause[:140]}..."

    # 2) Mail
    alert_msg, res_msg, mail_score = match_mail(error_text)
    mail_found = alert_msg is not None or res_msg is not None
    mail_summary = "No prior mail evidence found."
    mail_details = MailDetails()

    if mail_found:
        # prefer resolution content if present
        subj = None
        frm = None
        recv = None

        if res_msg:
            subj = res_msg.get("subject")
            frm = res_msg.get("from")
            recv = res_msg.get("received_at")
        elif alert_msg:
            subj = alert_msg.get("subject")
            frm = alert_msg.get("from")
            recv = alert_msg.get("received_at")

        mail_details = MailDetails(
            thread_id=(res_msg or alert_msg or {}).get("thread_id"),
            subject=subj,
            from_addr=frm,
            received_at=recv,
            alert_body=(alert_msg or {}).get("body"),
            resolution_body=(res_msg or {}).get("body"),
            related_incident_id=(res_msg or alert_msg or {}).get("related_incident_id"),
        )

        # short summary: try to pull "Root cause:" or "Fix:" lines from resolution
        preview = ""
        if mail_details.resolution_body:
            m = re.search(r"(Root cause:.*|Fix:.*|Recovery:.*)", mail_details.resolution_body, flags=re.IGNORECASE)
            preview = (m.group(1).strip() if m else mail_details.resolution_body.strip().splitlines()[0])
        else:
            preview = (mail_details.alert_body or "").strip().splitlines()[0] if mail_details.alert_body else ""

        mail_summary = f"Match: {mail_details.thread_id} • {preview[:160]}..."

    # 3) Server Logs
    log_details, log_score = match_server_logs(error_text, alert_msg)
    log_found = log_details is not None
    log_summary = "No matching server log found."
    if log_found and log_details:
        if log_details.file_exists:
            log_summary = f"Log found: {log_details.log_path} • excerpt available"
        else:
            log_summary = f"Log referenced: {log_details.log_path} • file not mounted in gateway"

    # 4) Fallback (only if all three missing)
    fallback_used = (not sn_found) and (not mail_found) and (not log_found)
    fb_block = BlockSummary(found=False, title="Fallback", summary="Not used.")
    fb_details = FallbackDetails(used=False)

    if fallback_used:
        if tavily_client:
            ans, sources = internet_fallback(error_text)
            if ans:
                fb_details = FallbackDetails(used=True, mode="internet", answer=ans, sources=sources)
                fb_block = BlockSummary(
                    found=True,
                    title="Fallback (Internet/Open-source)",
                    summary=(ans[:220] + ("..." if len(ans) > 220 else "")),
                    confidence=0.50
                )
            else:
                # Tavily configured but no answer returned
                gen = generic_fallback(error_text)
                fb_details = FallbackDetails(used=True, mode="generic", answer=gen, sources=[])
                fb_block = BlockSummary(found=True, title="Fallback (Generic)", summary=gen[:220] + "...", confidence=0.40)
        else:
            gen = generic_fallback(error_text)
            fb_details = FallbackDetails(used=True, mode="generic", answer=gen, sources=[])
            fb_block = BlockSummary(found=True, title="Fallback (Generic)", summary=gen[:220] + "...", confidence=0.40)

    # Response: summaries for Tab 1 + details for Tab 2
    return AnalyzeResp(
        servicenow=BlockSummary(
            found=sn_found,
            title="ServiceNow",
            summary=sn_summary,
            confidence=round(inc_score, 3) if sn_found else round(inc_score, 3),
        ),
        mail=BlockSummary(
            found=mail_found,
            title="Mail",
            summary=mail_summary,
            confidence=round(mail_score, 3) if mail_found else round(mail_score, 3),
        ),
        server_logs=BlockSummary(
            found=log_found,
            title="Server Logs",
            summary=log_summary,
            confidence=round(log_score, 3) if log_found else round(log_score, 3),
        ),
        fallback=fb_block,
        details={
            "servicenow": sn_details.model_dump(),
            "mail": mail_details.model_dump(),
            "server_logs": (log_details.model_dump() if log_details else {}),
            "fallback": fb_details.model_dump(),
        }
    )
