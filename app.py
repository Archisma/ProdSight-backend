import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from difflib import SequenceMatcher

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Aurora RAG stack (optional but kept, as you asked to keep old code style)
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# CrewAI + Tavily (optional / public search)
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from tavily import TavilyClient


# =============================================================================
# CONFIG
# =============================================================================
APP_DIR = Path(__file__).resolve().parent

# Evidence data files
INCIDENTS_FILE = APP_DIR / "incidents.json"
MAILBOX_FILE   = APP_DIR / "mailbox.json"

# 3 log roots (host locally; in prod you can point these to mounted paths)
JAVA_LOG_ROOT     = Path(os.environ.get("LOG_ROOT_JAVA", str(APP_DIR / "logs" / "java")))
AUTOSYS_LOG_ROOT  = Path(os.environ.get("LOG_ROOT_AUTOSYS", str(APP_DIR / "logs" / "autosys")))
ABINITIO_LOG_ROOT = Path(os.environ.get("LOG_ROOT_ABINITIO", str(APP_DIR / "logs" / "autosys" / "abinitio")))

# Similarity thresholds
INC_MIN_SCORE  = float(os.environ.get("INC_MIN_SCORE", "0.62"))
MAIL_MIN_SCORE = float(os.environ.get("MAIL_MIN_SCORE", "0.58"))

# LLM + Search keys (Render env vars)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# Demo remote simulation settings
DEMO_REMOTE = os.environ.get("DEMO_REMOTE", "true").lower() == "true"
BASTION_HOST = os.environ.get("BASTION_HOST", "bastion.prod.local")
JAVA_HOST    = os.environ.get("JAVA_HOST", "java-prod-01")
AUTOSYS_HOST = os.environ.get("AUTOSYS_HOST", "autosys-prod-01")

MID_SERVER   = os.environ.get("MID_SERVER", "mid-prod-01")
SNOW_INSTANCE = os.environ.get("SNOW_INSTANCE", "instance.service-now.com")

if not GEMINI_API_KEY:
    print("❌ Missing GEMINI_API_KEY (or GOOGLE_API_KEY). Set it in Render Environment Variables.")


# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(title="ProdSight Backend", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later to your Cloudflare Pages domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REQUEST / RESPONSE MODELS
# =============================================================================
class SearchReq(BaseModel):
    query: str

class PublicSearchReq(BaseModel):
    topic: str

class AnalyzeReq(BaseModel):
    error_text: str

class BlockSummary(BaseModel):
    found: bool
    title: str
    summary: str = ""
    confidence: float = 0.0

class AnalyzeResp(BaseModel):
    servicenow: BlockSummary
    mail: BlockSummary
    server_logs: BlockSummary
    prodsight_answer: str
    details: Dict[str, Any]


# =============================================================================
# LLM (Gemini)
# =============================================================================
llm = ChatGoogleGenerativeAI(
    model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    temperature=0,
    google_api_key=GEMINI_API_KEY
)


# =============================================================================
# OPTIONAL: Aurora RAG endpoints (kept from old project)
# You can remove this entire section if not required for ProdSight.
# =============================================================================
AURORA_DATA_DIR = "data"
PDF_PATH  = os.path.join(AURORA_DATA_DIR, "aurora_systems_full_policy_manual.pdf")
DOCX_PATH = os.path.join(AURORA_DATA_DIR, "aurora_systems_long_term_strategy.docx")
XLSX_PATH = os.path.join(AURORA_DATA_DIR, "aurora_systems_operational_financials.xlsx")

docs: List[Document] = []

if os.path.exists(PDF_PATH):
    docs += PyPDFLoader(PDF_PATH).load()
if os.path.exists(DOCX_PATH):
    docs += Docx2txtLoader(DOCX_PATH).load()
if os.path.exists(XLSX_PATH):
    xls = pd.ExcelFile(XLSX_PATH)
    for sheet in xls.sheet_names:
        df = pd.read_excel(XLSX_PATH, sheet_name=sheet)
        text = f"Excel Sheet: {sheet}\n\n" + df.to_string(index=False)
        docs.append(Document(page_content=text, metadata={"source": XLSX_PATH, "sheet": sheet}))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
split_documents = text_splitter.split_documents(docs) if docs else []

retriever = None
rag_chain = None

if GEMINI_API_KEY and split_documents:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=GEMINI_API_KEY
    )
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

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


# =============================================================================
# OPTIONAL: Tavily public search (kept from old code)
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
    return str(crew.kickoff())


# =============================================================================
# Helper: JSON load
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


# =============================================================================
# Helper: similarity
# =============================================================================
def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _sim(a: str, b: str) -> float:
    a2, b2 = _norm(a), _norm(b)
    if not a2 or not b2:
        return 0.0
    return SequenceMatcher(None, a2, b2).ratio()


# =============================================================================
# ServiceNow matching (from incidents.json)
# =============================================================================
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


# =============================================================================
# Mail matching (from mailbox.json)
# =============================================================================
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


# =============================================================================
# Server log search: portable grep across 3 roots
# =============================================================================
def grep_in_logs(
    root: Path,
    query: str,
    max_files: int = 120,
    max_matches: int = 30,
    max_line_len: int = 300
) -> Dict[str, Any]:
    out = {
        "root": str(root),
        "found": False,
        "matches": []
    }

    if not root.exists() or not root.is_dir():
        out["error"] = "log_root_missing"
        return out

    q = (query or "").strip()
    if not q:
        out["error"] = "empty_query"
        return out

    q_low = q.lower()
    scanned = 0
    matched = 0

    patterns = ["*.log", "*.out", "*.txt"]
    files: List[Path] = []
    for p in patterns:
        files.extend(root.rglob(p))

    files = sorted(files)[:max_files]

    for fp in files:
        try:
            scanned += 1
            with fp.open("r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, start=1):
                    if q_low in line.lower():
                        out["found"] = True
                        matched += 1
                        out["matches"].append({
                            "file": str(fp),
                            "line_no": i,
                            "line": (line.strip()[:max_line_len] + ("…" if len(line.strip()) > max_line_len else ""))
                        })
                        if matched >= max_matches:
                            out["truncated"] = True
                            out["scanned_files"] = scanned
                            out["match_count"] = matched
                            return out
        except Exception:
            continue

    out["scanned_files"] = scanned
    out["match_count"] = matched
    return out


# =============================================================================
# Demo meta: SSH via bastion + MID server
# =============================================================================
def build_remote_meta(error_text: str) -> dict:
    # Short signature for demo grep command readability
    sig = (error_text.strip().splitlines()[0] if error_text else "error_signature")[:120]
    sig = sig.replace('"', '\\"')

    return {
        "server_logs_meta": {
            "mode": "ssh_via_bastion" if DEMO_REMOTE else "local",
            "bastion": BASTION_HOST,
            "targets": [
                {"name": JAVA_HOST, "type": "java", "path_hint": "/var/log/java"},
                {"name": AUTOSYS_HOST, "type": "autosys", "path_hint": "/var/log/autosys"},
                {"name": AUTOSYS_HOST, "type": "abinitio", "path_hint": "/var/log/abinitio"},
            ],
            "demo_commands": [
                f"ssh prodsight@{BASTION_HOST}",
                f"ssh {JAVA_HOST}   # JAVA",
                f'grep -R "{sig}" /var/log/java',
                f"ssh {AUTOSYS_HOST} # AUTOSYS + ABINITIO",
                f'grep -R "{sig}" /var/log/autosys',
                f'grep -R "{sig}" /var/log/abinitio',
            ]
        },
        "servicenow_meta": {
            "mode": "mid_server_integration",
            "instance": SNOW_INSTANCE,
            "mid_server": MID_SERVER,
            "demo_flow": [
                f"ProdSight -> MID Server ({MID_SERVER})",
                f"MID Server -> ServiceNow ({SNOW_INSTANCE})",
                "Query: recent INCs + similar error signatures",
                "Return: INC list + previous error + resolution steps"
            ]
        }
    }


# =============================================================================
# ENDPOINTS
# =============================================================================
@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "ProdSight – Production Intelligence for Faster Resolution",
        "proprietary": "TCS",
        "gemini_model": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        "incidents_file_exists": INCIDENTS_FILE.exists(),
        "mailbox_file_exists": MAILBOX_FILE.exists(),
        "log_roots": {
            "java": str(JAVA_LOG_ROOT),
            "autosys": str(AUTOSYS_LOG_ROOT),
            "abinitio": str(ABINITIO_LOG_ROOT),
        },
        "demo_remote": DEMO_REMOTE,
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

    # Summary should be one-liner count in UI; backend can still provide a summary string
    sn_summary = "Total matching incidents: 1" if sn_found else "Total matching incidents: 0"

    # --- Mail ---
    alert_msg, res_msg, mail_score = match_mail(error_text)
    mail_found = bool(alert_msg or res_msg)
    mail_summary = "Total matching threads: 1" if mail_found else "Total matching threads: 0"

    # --- Server Logs (3 roots) ---
    java_hit = grep_in_logs(JAVA_LOG_ROOT, error_text)
    autosys_hit = grep_in_logs(AUTOSYS_LOG_ROOT, error_text)
    abinitio_hit = grep_in_logs(ABINITIO_LOG_ROOT, error_text)

    any_found = bool(java_hit.get("found") or autosys_hit.get("found") or abinitio_hit.get("found"))

    if any_found:
        parts = []
        if java_hit.get("found"):
            parts.append(f"JAVA({len(java_hit.get('matches', []))} hits)")
        if autosys_hit.get("found"):
            parts.append(f"AUTOSYS({len(autosys_hit.get('matches', []))} hits)")
        if abinitio_hit.get("found"):
            parts.append(f"ABINITIO({len(abinitio_hit.get('matches', []))} hits)")
        logs_summary = "Matches found: " + ", ".join(parts)
        logs_conf = 0.80
    else:
        logs_summary = "No matching error found in JAVA/AUTOSYS/ABINITIO logs."
        logs_conf = 0.0

    # --- Build evidence bundle for Details tab ---
    evidence: Dict[str, Any] = {
        "servicenow": inc or {},
        "mail": {"alert": alert_msg or {}, "resolution": res_msg or {}},
        "server_logs": {
            "java": java_hit,
            "autosys": autosys_hit,
            "abinitio": abinitio_hit,
        }
    }

    # Add demo “remote session” metadata
    meta = build_remote_meta(error_text)
    evidence["server_logs_meta"] = meta["server_logs_meta"]
    evidence["servicenow_meta"] = meta["servicenow_meta"]

    # --- Gemini: write final answer (evidence-first, no hallucinations) ---
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

INPUT ERROR:
{error_text}

EVIDENCE (JSON):
{json.dumps(evidence, indent=2)}
""".strip()

    try:
        prodsight_answer = llm.invoke(prodsight_prompt).content
    except Exception as e:
        prodsight_answer = f"(LLM unavailable) Evidence summary only. Error: {str(e)}"

    return AnalyzeResp(
        servicenow=BlockSummary(found=sn_found, title="ServiceNow", summary=sn_summary, confidence=round(float(inc_score), 3)),
        mail=BlockSummary(found=mail_found, title="Mail", summary=mail_summary, confidence=round(float(mail_score), 3)),
        server_logs=BlockSummary(found=any_found, title="Server Logs", summary=logs_summary, confidence=logs_conf),
        prodsight_answer=prodsight_answer,
        details=evidence
    )
