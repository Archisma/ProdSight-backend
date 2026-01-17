ProdSight – Production Intelligence for Faster Resolution
(Proprietary solution by TCS – client showcase)

=========================================================
1) What is ProdSight?
=========================================================
ProdSight is a Production Support (L3) intelligence gateway that helps teams resolve incidents faster by
correlating evidence from:
  1) ServiceNow (INC tickets)
  2) Mailbox (historical alerts + resolution emails)
  3) Server Logs (log path + optional excerpts if logs are mounted)

It follows an "internal-evidence first" approach:
- If ServiceNow/Mail/Logs have matches, it shows structured results.
- If all are missing, it falls back to an open-source/internet explanation (optional) or a generic L3 triage checklist.

=========================================================
2) Architecture (Render + Cloudflare Pages)
=========================================================
Backend API (FastAPI)  -> Render (public HTTPS endpoint)
Frontend UI (HTML/JS)  -> Cloudflare Pages (static site)

The Cloudflare UI calls the Render API:
  POST /analyze
  GET  /health

=========================================================
3) Files included
=========================================================
Backend:
- app.py                  (ProdSight API gateway)
- incidents.json          (ServiceNow mock data for demo)
- mailbox.json            (Mailbox mock data for demo)
- requirements.txt

Frontend:
- index.html              (ProdSight UI – Summary + Details tabs)

Optional:
- server_logs/            (mounted logs folder used for log excerpts)
  Example path mapping:
    If a log path is: /var/log/aurora/prod/billing-api/2026-01-16.log
    Then place it at:
      server_logs/var/log/aurora/prod/billing-api/2026-01-16.log

=========================================================
4) API Endpoints
=========================================================
GET /health
- Returns service status, whether demo data files exist, and whether Tavily fallback is configured.

POST /analyze
Request JSON:
{
  "error_text": "paste stacktrace/logs here"
}

Response JSON:
- Summary blocks for UI Tab 1:
  - servicenow: {found, summary, confidence}
  - mail: {found, summary, confidence}
  - server_logs: {found, summary, confidence}
  - fallback: {found, summary, confidence}

- Detailed payload for UI Tab 2:
  details: {
    "servicenow": {...},
    "mail": {...},
    "server_logs": {...},
    "fallback": {...}
  }

=========================================================
5) Environment Variables (Render)
=========================================================
Required: none (demo mode uses local JSON)

Optional:
- LOG_ROOT
  Default: ./server_logs
  Description: Folder where server logs are mounted/mirrored.

- TAVILY_API_KEY
  If set, enables internet/open-source fallback when internal evidence is missing.

Similarity thresholds (optional tuning):
- INC_MIN_SCORE   (default 0.62)
- MAIL_MIN_SCORE  (default 0.58)
- LOG_MIN_SCORE   (default 0.55)

=========================================================
6) Run Locally
=========================================================
1) Install dependencies:
   pip install -r requirements.txt

2) Start backend:
   uvicorn app:app --host 0.0.0.0 --port 10000

3) Open the UI:
   - If using a local static server, serve index.html
   - Update API_BASE in index.html to point to your backend:
     e.g. http://localhost:10000

=========================================================
7) Deploy (Recommended)
=========================================================
Backend (Render):
- Create a new Web Service
- Build Command: pip install -r requirements.txt
- Start Command: uvicorn app:app --host 0.0.0.0 --port 10000
- Add env vars if needed (TAVILY_API_KEY, LOG_ROOT)

Frontend (Cloudflare Pages):
- Create Pages project
- Upload/commit index.html
- Set API_BASE in index.html to the Render backend URL

=========================================================
8) Security / OAuth Notes
=========================================================
This version uses mock ServiceNow + mailbox JSON for demo.
For production, ServiceNow and Mail should be integrated via secure auth:
- OAuth2 / SSO integration (TCS client environment)
- Restrict CORS to the Cloudflare Pages domain
- Backend-side auth validation (no client-side gates)

=========================================================
9) Branding
=========================================================
Client-facing name: ProdSight
Tagline: Production Intelligence for Faster Resolution
Ownership: Proprietary solution by TCS (client showcase)
