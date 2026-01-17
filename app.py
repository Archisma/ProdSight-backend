import os
import re
import json
import time
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from difflib import SequenceMatcher

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# CrewAI + Tavily
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from tavily import TavilyClient


# =============================================================================
# CONFIG
# =============================================================================
# Aurora demo data folder (kept as-is)
AURORA_DATA_DIR = "data"

PDF_PATH  = os.path.join(AURORA_DATA_DIR, "aurora_systems_full_policy_manual.pdf")
DOCX_PATH = os.path.join(AURORA_DATA_DIR, "aurora_systems_long_term_strategy.docx")
XLSX_PATH = os.path.join(AURORA_DATA_DIR, "aurora_systems_operational_financials.xlsx")

# ProdSight evidence files (same folder as app.py)
APP_DIR = Path(__file__).resolve().parent
INCIDENTS_FILE = APP_DIR / "incidents.json"
MAILBOX_FILE   = APP_DIR / "mailbox.json"

# Optional server logs mount
LOG_ROOT = Path(os.environ.get("LOG_ROOT", str(APP_DIR / "server_logs")))

# Similarity thresholds (tune later)
INC_MIN_SCORE  = float(os.environ.get("INC_MIN_SCORE", "0.62"))
MAIL_MIN_SCORE = float(os.environ.get("MAIL_MIN_SCORE", "0.58"))

# Render env vars
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if not GEMINI_API_KEY:
    print("❌ Missing GEMINI_API_KEY (or GOOGLE_API_KEY). Set it in Render Environment Variables.")


# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(title="ProdSight Backend", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your Cloudflare Pages domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REQUEST MODELS
# =============================================================================
class SearchReq(BaseModel):
    query: str

class PublicSearchReq(BaseModel):
    topic: str

class AnalyzeReq(BaseModel):
    error_text: str


# =============================================================================
# ProdSight Response Models
# =============================================================================
class BlockSummary(BaseModel):
    found: bool
    title: str
    summary: str = ""
    confidence: float = 0.0

class AnalyzeResp(BaseModel):
    servicenow: BlockSummary
    mail: BlockSummary
    server_logs: BlockSummary
    fallback: BlockSummary
    prodsight_answer: str
    details: Dict[str, Any]


# =============================================================================
# 1) LOAD DOCUMENTS (Aurora RAG) - kept for your existing demo
# =============================================================================
docs: List[Document] = []

# PDF
if os.path.exists(PDF_PATH):
    pdf_docs = PyPDFLoader(PDF_PATH).load()
    docs += pdf_docs
    print(f"✅ PDF loaded successfully: {os.path.basename(PDF_PATH)} (pages={len(pdf_docs)})")
else:
    print(f"⚠️ PDF not found: {PDF_PATH}")

# DOCX
if os.path.exists(DOCX_PATH):
    docx_docs = Docx2txtLoader(DOCX_PATH).load()
    docs += docx_docs
    print(f"✅ Word loaded successfully: {os.path.basename(DOCX_PATH)} (sections={len(docx_docs)})")
else:
    print(f"⚠️ Word not found: {DOCX_PATH}")

# XLSX
if os.path.exists(XLSX_PATH):
    xls = pd.ExcelFile(XLSX_PATH)
    sheet_count = 0
    for sheet in xls.sheet_names:
        df = pd.read_excel(XLSX_PATH, sheet_name=sheet)
        text = f"Excel Sheet: {sheet}\n\n" + df.to_string(index=False)
        docs.append(Document(page_content=text, metadata={"source": XLSX_PATH, "sheet": sheet}))
        sheet_count += 1
    print(f"✅ Excel loaded successfully: {os.path.basename(XLSX_PATH)} (sheets={sheet_count})")
else:
    print(f"⚠️ Excel not found: {XLSX_PATH}")

print(f"📄 Aurora RAG docs loaded: {len(docs)} total")


# =============================================================================
# 2) SPLIT / CHUNK (Aurora)
# =============================================================================
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
split_documents = text_splitter.split_documents(docs) if docs else []
print(f"✅ Chunking completed: {len(split_documents)} chunks")


# =============================================================================
# 3) EMBEDDINGS + VECTORSTORE + RAG CHAIN (Aurora)
# =============================================================================
# NOTE: If you deploy without Aurora docs, the /search endpoint will still run,
# but it will respond "I don't know" because there is no context.
embeddings = None
vectorstore = None
retriever = None
rag_chain = None

if GEMINI_API_KEY and split_documents:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=GEMINI_API_KEY
    )
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatGoogleGenerativeAI(
    model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    temperature=0,
    google_api_key=GEMINI_API_KEY
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Answer using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
""")

if retriever:
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

print("✅ Gemini LLM ready")


# =============================================================================
# CrewAI + Tavily Tool (Public Search) - kept
# =============================================================================
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

class TavilySearchTool(BaseTool):
    name: str = "tavily_web_search"
    description: str = "Search the internet using Tavily and return short results with source URLs."

    def _run(self, query: str) -> str:
        if not tavily_client:
            return "Tavily is not configured (missing TAVILY_API_KEY)."

        result = tavily_client.search(
            query=query,
            max_results=6,
            include_answer=True,
            include_sources=True
        )

        answer = result.get("answer", "")
        sources = result.get("results", [])

        lines = []
        lines.append("WEB ANSWER:")
        lines.append(answer if answer else "(no answer returned)")
        lines.append("\nSOURCES:")
        for i, s in enumerate(sources, start=1):
            lines.append(f"{i}) {s.get('title','No title')}\n   {s.get('url','')}")
        return "\n".join(lines).strip()

tavily_tool = TavilySearchTool()

def run_crewai_public(topic: str) -> str:
    researcher = Agent(
        role="Internet Researcher",
        goal="Find accurate, concise information with sources.",
        backstory="You are a careful researcher who cites sources and avoids hallucinations.",
        tools=[tavily_tool],
        verbose=False
    )

    task = Task(
        description=f"Research this topic and provide a short answer with sources:\n{topic}",
        agent=researcher
    )

    crew = Crew(agents=[researcher], tasks=[task], verbose=False)
    result = crew.kickoff()
    return str(result)


# =============================================================================
# ProdSight helpers (ServiceNow + Mail + Server Logs)
# =============================================================================
def _safe_load_json(path: Path, default: Any):
    try:
        if not path.exists():
            return default
        raw = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not raw:
            return default
        return json.loads(raw)
    except Exception:
        return default

def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _sim(a: str, b: str) -> float:
    a2, b2 = _norm(a), _norm(b)
    if not a2 or not b2:
        return 0.0
    return SequenceMatcher(None, a2, b2).ratio()

def extract_log_path(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"LogPath:\s*([^\s]+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"(/var/log/[A-Za-z0-9_\-./]+)", text)
    if m2:
        return m2.group(1).strip()
    return None

def read_log_excerpt(log_path: str, max_lines: int = 80) -> Tuple[bool, Optional[str]]:
    if not log_path:
        return False, None
    rel = log_path.lstrip("/")
    candidate = LOG_ROOT / rel
    if not candidate.exists() or not candidate.is_file():
        return False, None
    try:
        lines = candidate.read_text(encoding="utf-8", errors="ignore").splitlines()
        tail = lines[-max_lines:] if len(lines) > max_lines else lines
        return True, "\n".join(tail)
    except Exception:
        return True, None

def match_servicenow(error_text: str) -> Tuple[Optional[Dict[str, Any]], float]:
    incidents = _safe_load_json(INCIDENTS_FILE, default=[])
    if not isinstance(incidents, list):
        return None, 0.0

    best = None
    best_score = 0.0
    for inc in incidents:
        sig = (inc.get("error_signature") or "")
        sc = _sim(error_text, sig)
        if sc > best_score:
            best_score = sc
            best = inc

    if best and best_score >= INC_MIN_SCORE:
        return best, best_score
    return None, best_score

def load_mail_messages() -> List[Dict[str, Any]]:
    mb = _safe_load_json(MAILBOX_FILE, default={})
    folders = (mb or {}).get("folders", {})
    msgs: List[Dict[str, Any]] = []
    if isinstance(folders, dict):
        for _, arr in folders.items():
            if isinstance(arr, list):
                msgs.extend(arr)
    return msgs

def group_threads(messages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    threads: Dict[str, List[Dict[str, Any]]] = {}
    for m in messages:
        tid = m.get("thread_id") or f"MSG-{m.get('id')}"
        threads.setdefault(tid, []).append(m)
    for _, arr in threads.items():
        arr.sort(key=lambda x: x.get("received_at") or "")
    return threads

def match_mail(error_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], float]:
    msgs = load_mail_messages()
    if not msgs:
        return None, None, 0.0

    threads = group_threads(msgs)

    best_tid = None
    best_score = 0.0
    for tid, arr in threads.items():
        tbest = 0.0
        for m in arr:
            text = f"{m.get('subject','')}\n{m.get('body','')}"
            tbest = max(tbest, _sim(error_text, text))
        if tbest > best_score:
            best_score = tbest
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

    if not alert and arr:
        alert = arr[0]
    if not resolution and len(arr) > 1:
        resolution = arr[-1]

    return alert, resolution, best_score

def internet_fallback(error_text: str) -> Tuple[str, List[Dict[str, str]]]:
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
    sources = [{"title": (r.get("title") or "source").strip(), "url": (r.get("url") or "").strip()}
               for r in (result.get("results") or [])]
    return answer, sources

def generic_fallback(error_text: str) -> str:
    first_line = (error_text.strip().splitlines()[:1] or [""])[0]
    return (
        "No internal evidence found in ServiceNow, Mail, or Server Logs.\n\n"
        "Suggested L3 triage (generic):\n"
        "1) Identify the failing component/service and the first exception line.\n"
        "2) Check recent deployments/config changes for that service.\n"
        "3) Correlate time window in logs/metrics (latency, errors, retries, GC, DB locks).\n"
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
        "service": "ProdSight – Production Intelligence for Faster Resolution",
        "proprietary": "TCS",
        "incidents_file_exists": INCIDENTS_FILE.exists(),
        "mailbox_file_exists": MAILBOX_FILE.exists(),
        "log_root": str(LOG_ROOT),
        "tavily_configured": bool(tavily_client),
        "gemini_model": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

@app.post("/search")
def search(req: SearchReq):
    if not rag_chain:
        return {"answer": "I don't know (Aurora RAG context not loaded)."}
    out = rag_chain.invoke(req.query)
    return {"answer": getattr(out, "content", str(out))}

@app.post("/public_search")
def public_search(req: PublicSearchReq):
    if not TAVILY_API_KEY:
        return {"answer": "Tavily not configured (missing TAVILY_API_KEY)."}
    return {"answer": run_crewai_public(req.topic)}

@app.post("/analyze", response_model=AnalyzeResp)
def analyze(req: AnalyzeReq):
    error_text = (req.error_text or "").strip()
    if not error_text:
        raise HTTPException(status_code=400, detail="Please provide error_text.")

    # --- ServiceNow ---
    inc, inc_score = match_servicenow(error_text)
    sn_found = inc is not None
    sn_summary = "Not found in ServiceNow."
    if sn_found:
        root_cause = (inc.get("root_cause") or "")
        sn_summary = f"Match {inc.get('incident_id','(no-id)')}: {root_cause[:160]}"

    # --- Mail ---
    alert_msg, res_msg, mail_score = match_mail(error_text)
    mail_found = bool(alert_msg or res_msg)
    mail_summary = "Not found in Mail."
    if mail_found:
        chosen = res_msg or alert_msg or {}
        preview = ""
        body = (res_msg or {}).get("body") or ""
        if body:
            m = re.search(r"(Root cause:.*|Fix:.*|Recovery:.*)", body, flags=re.IGNORECASE)
            preview = (m.group(1).strip() if m else body.strip().splitlines()[0])
        else:
            preview = ((alert_msg or {}).get("body") or "").strip().splitlines()[0] if alert_msg else ""
        mail_summary = f"Match {chosen.get('thread_id','(no-thread)')}: {preview[:160]}"

    # --- Server Logs ---
    log_path = None
    if alert_msg:
        log_path = extract_log_path(alert_msg.get("body", ""))
    if not log_path:
        log_path = extract_log_path(error_text)

    log_found = bool(log_path)
    log_exists, excerpt = (False, None)
    if log_found:
        log_exists, excerpt = read_log_excerpt(log_path)

    logs_summary = "Not found in Server Logs."
    if log_found:
        logs_summary = f"Log referenced: {log_path} (file not mounted)"
        if log_exists:
            logs_summary = f"Log available: {log_path} (excerpt ready)"

    # --- Fallback only if all missing ---
    fallback_used = (not sn_found) and (not mail_found) and (not log_found)
    fb_summary = "Not used."
    fb_found = False
    fb_conf = 0.0

    fb_details: Dict[str, Any] = {"used": False, "mode": "none", "answer": "", "sources": []}
    if fallback_used:
        if tavily_client:
            ans, sources = internet_fallback(error_text)
            if ans:
                fb_found = True
                fb_conf = 0.50
                fb_summary = ans[:220] + ("..." if len(ans) > 220 else "")
                fb_details = {"used": True, "mode": "internet", "answer": ans, "sources": sources}
            else:
                gen = generic_fallback(error_text)
                fb_found = True
                fb_conf = 0.40
                fb_summary = gen[:220] + "..."
                fb_details = {"used": True, "mode": "generic", "answer": gen, "sources": []}
        else:
            gen = generic_fallback(error_text)
            fb_found = True
            fb_conf = 0.40
            fb_summary = gen[:220] + "..."
            fb_details = {"used": True, "mode": "generic", "answer": gen, "sources": []}

    # --- Evidence bundle for Details tab ---
    evidence = {
        "servicenow": inc or {},
        "mail": {"alert": alert_msg or {}, "resolution": res_msg or {}},
        "server_logs": {"log_path": log_path, "file_exists": bool(log_exists), "excerpt": excerpt or ""},
        "fallback": fb_details,
    }

    # --- Gemini: produce final answer from evidence (no hallucinations) ---
    prodsight_prompt = f"""
You are ProdSight – Production Intelligence for Faster Resolution.
Proprietary solution by TCS (client showcase).

RULES:
- Do NOT invent ticket IDs, log paths, timestamps, owners, or fixes.
- Use ONLY the evidence provided below.
- If something is missing, clearly say "Not found".

Produce:
1) Summary (2-4 sentences)
2) Likely root cause (bullets; can be Unknown)
3) Recommended actions (numbered, actionable)
4) Evidence status:
   - ServiceNow: Found/Not Found
   - Mail: Found/Not Found
   - Server Logs: Found/Not Found
   - Fallback: Used/Not Used

INPUT ERROR:
{error_text}

EVIDENCE (JSON):
{json.dumps(evidence, indent=2)}
""".strip()

    try:
        # Uses your existing LangChain Gemini LLM instance
        prodsight_answer = llm.invoke(prodsight_prompt).content
    except Exception as e:
        prodsight_answer = f"(LLM unavailable) Evidence summary only. Error: {str(e)}"

    return AnalyzeResp(
        servicenow=BlockSummary(found=sn_found, title="ServiceNow", summary=sn_summary, confidence=round(float(inc_score), 3)),
        mail=BlockSummary(found=mail_found, title="Mail", summary=mail_summary, confidence=round(float(mail_score), 3)),
        server_logs=BlockSummary(found=log_found, title="Server Logs", summary=logs_summary, confidence=0.70 if log_found else 0.0),
        fallback=BlockSummary(found=fb_found, title="Fallback", summary=fb_summary, confidence=fb_conf),
        prodsight_answer=prodsight_answer,
        details=evidence
    )
