# AI-Hospital Copilot Playbook

## Architecture snapshot
- `backend/main.py` creates the DB schema, seeds static doctors, applies CORS, and mounts both auth routers (`routers/users.py`, `routers/oauth.py`) plus the LangGraph router from `backend/api.py`.
- `backend/AI_hospital.py` owns `AgentState`, global `patient_info`, LangGraph nodes, and all tool bindings; every state mutation or new route must be reflected here.
- `frontend/src/ui/App.tsx` is a Vite SPA that maintains chat, tool telemetry, and active node labels off SSE events; breaking the event contract instantly breaks the UI.
- Vector RAG stores live in `backend/vector_stores/{specialty}` and are loaded by `Knowledge_notebooks/initialize_rag.py` with `BAAI/bge-large-en-v1.5` embeddings on CPU.

## LangGraph flow rules
- GP node must: greet → ask via `ask_user` (exactly one question per call) → call `Patient_data_report` once demographics + key symptoms are known → emit only the canonical specialist name (router key).
- `AgentState` keeps parallel transcripts (`messages`, `specialist_messages`, helper streams) plus `patho_QnA`, `radio_QnA`, `next_agent`, `agent_order`, `current_report`, and `patient_id`. Omit any of these when adding nodes and the downstream routers will crash.
- Specialist routers append helper prompts into `patho_QnA` / `radio_QnA`, push their own name onto `next_agent`, and watch for `Final Report:` to terminate with `END`.
- Helpers run as standard agents but must always pop the caller from `next_agent` so their findings route back to the correct specialist thread.

## Tool semantics (all defined in `AI_hospital.py`)
- `ask_user` never actually executes; `backend/api.py` intercepts calls via `ASK_NODES` and emits an `ask_user` SSE event. Missing a node name here causes the graph to hang.
- `Patient_data_report(data, state)` persists GP triage into `Consultation` (status `Active`) using the injected `patient_id`. Call once per patient session or DB writes will fail.
- `add_report(report, state)` inspects the live consultation: helper notes create `LabOrder`/`LabResult` rows, while reports containing “Final Report”/“Diagnosis” close the consultation and create a `MedicalReport` entry.
- `VectorRAG_Retrival(query, agent)` requires the canonical specialist label (router strings work). It pulls five docs from the matching Chroma store; reformulate the query instead of looping infinitely.

## API + auth contract
- `/api/graph/start/stream` and `/api/graph/resume/stream` only work with a valid JWT token passed as a query parameter; the token supplies `patient_id`, which is injected into LangGraph state.
- EventSource stream order is `thread` → `message` / `tool` repeats → `ask_user` or `final`. `_chunk_to_payload` normalizes all values; keep its shape stable if you change state keys.
- When extending the graph, add every new `*_AskUser` node to `ASK_NODES`, and ensure `_speaker_for_key` returns a label consumed by the frontend speaker map.

## Frontend expectations
- The SPA deduplicates assistant messages by raw string match; if you change backend formatting, also adjust the duplicate guard in `App.tsx`.
- `pendingAsk` UI closes the SSE stream as soon as `ask_user` fires; `resume/stream` must replay the latest context quickly or the UX feels frozen.
- Configure `VITE_API_BASE` when the backend isn’t proxied; otherwise `/api/...` is assumed.

## Developer workflows & ops
- Backend: `cd backend && uv sync && uv run uvicorn backend.main:app --reload --port 8000`. `.env` must contain `GEMINI_API_KEY` + `TAVILY_API_KEY` or tool nodes will raise at import time.
- Frontend: `cd frontend && npm install && npm run dev`; adjust Vite proxy or `VITE_API_BASE` to hit the backend port 8000.
- Vector refresh: drop PDFs under `backend/Knowledge Base/{specialty}` and rerun `backend/Knowledge_notebooks/vector_rag.ipynb`; `VectorRAG_initialize()` expects those directories to exist before FastAPI starts.
- DB seeding: `backend/main.py` seeds static doctors on import. Avoid heavy work in module scope elsewhere or server startup slows dramatically.
