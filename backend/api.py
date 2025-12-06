from fastapi import APIRouter, HTTPException, Depends, status
from uuid import uuid4
from typing import Optional, List
from sqlalchemy.orm import Session # <--- NEW

from .AI_hospital import myapp
from . import database, models, oauth2 # <--- NEW IMPORTS
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from sse_starlette.sse import EventSourceResponse
import json

router = APIRouter()

# ---------------------------
# Helpers
# ---------------------------
ASK_NODES = {
    "GP_AskUser", "Ophthal_AskUser", "Pedia_AskUser", "Ortho_AskUser",
    "Dermat_AskUser", "ENT_AskUser", "Gynec_AskUser", "Psych_AskUser",
    "IntMed_AskUser", "Patho_AskUser", "Radio_AskUser",
}

def _make_config(thread_id: str):
    return {"configurable": {"thread_id": thread_id}}

# UPDATED: Now accepts patient_id
def _initial_inputs(user_text: str, patient_id: int): 
    base_human = HumanMessage(content=user_text)
    return {
        "messages": [base_human],
        "specialist_messages": [base_human],
        "patho_messages": [HumanMessage(content="Generate some test based on status of Pathology status")],
        "radio_messages": [HumanMessage(content="Generate some report based on status of Radiology status")],
        "patho_QnA": [],
        "radio_QnA": [],
        "next_agent": [],
        "agent_order": [],
        "current_report": [],
        "current_agent": "GP",
        "patient_id": patient_id, # <--- INJECTING ID HERE
    }

# ... (Keep _extract_ask_question, _inject_user_reply_as_tool_message, 
#      _last_assistant_text, _speaker_for_key, _chunk_to_payload, 
#      _new_tool_calls exactly as they were) ...

# ---------------------------
# Endpoints
# ---------------------------

@router.get("/graph/start/stream")
async def start_graph_stream(
    message: str, 
    token: str, # <--- NEW: Accept token from URL
    db: Session = Depends(database.get_db) # <--- NEW: DB Session
):
    # 1. Manually Verify Token (Since it's in URL, not Header)
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # This verifies the signature and expiration
    token_data = oauth2.verify_access_token(token, credentials_exception)
    
    # 2. Get the Patient ID
    # We trust the token's ID, but you can also double check DB if you want:
    # patient = db.query(models.Patient).filter(models.Patient.patient_id == token_data.id).first()
    
    patient_id = int(token_data.id)

    thread_id = str(uuid4())
    config = _make_config(thread_id)
    
    # 3. Pass patient_id to the graph inputs
    inputs = _initial_inputs(message, patient_id)

    async def event_gen():
        yield {"event": "thread", "data": json.dumps({"thread_id": thread_id})}
        current_agent = "GP"
        seen_tool_ids: set = set()
        
        # ... (Rest of the loop is identical to previous code) ...
        async for chunk in myapp.astream(inputs, config, stream_mode="values"):
            for tc in _new_tool_calls(chunk, seen_tool_ids):
                yield {"event": "tool", "data": json.dumps({"thread_id": thread_id, **tc})}
            payload = _chunk_to_payload(chunk)
            if payload:
                agent_update = payload.get("current_agent") or chunk.get("current_agent")
                if agent_update:
                    current_agent = agent_update
                elif current_agent and "content" in payload:
                    payload.setdefault("current_agent", current_agent)
                yield {"event": "message", "data": json.dumps({"thread_id": thread_id, **payload})}
        
        # ... (Final state checks logic is identical) ...
        state = myapp.get_state(config)
        state_values = state.values or {}
        next_nodes = set(state.next or [])
        current_agent = state_values.get("current_agent", current_agent)
        
        if next_nodes & ASK_NODES:
            question = _extract_ask_question(state_values)
            ask_payload = {"thread_id": thread_id}
            if question:
                ask_payload["question"] = question
            if current_agent:
                ask_payload["current_agent"] = current_agent
                ask_payload["speaker"] = current_agent
            yield {"event": "ask_user", "data": json.dumps(ask_payload)}
        else:
            final = _last_assistant_text(state_values)
            final_payload = {"thread_id": thread_id, "message": final}
            if current_agent:
                final_payload["current_agent"] = current_agent
            yield {"event": "final", "data": json.dumps(final_payload)}

    return EventSourceResponse(event_gen())

@router.get("/graph/resume/stream")
async def resume_graph_stream(
    thread_id: str, 
    user_reply: str,
    token: str # <--- NEW: Verify token here too for security!
):
    # 1. Security Check
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
    )
    oauth2.verify_access_token(token, credentials_exception)
    
    # ... (Rest of function is identical to previous code) ...
    # config = _make_config(thread_id)
    # state = myapp.get_state(config)
    # ...