from fastapi import APIRouter, HTTPException, Depends, status
from uuid import uuid4
from typing import Optional, List
from sqlalchemy.orm import Session 

from .AI_hospital import myapp
from . import database, models, oauth2 
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

def _initial_inputs(user_text: str, patient_id: int):
    # Initialize all streams to keep graph happy
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
        "patient_id": patient_id, 
    }

def _extract_ask_question(state_values: dict) -> Optional[str]:
    """Find the ask_user tool call and return its 'question' argument."""
    for key in ("messages", "specialist_messages", "patho_messages", "radio_messages"):
        msgs: List = state_values.get(key) or []
        if not msgs:
            continue
        last = msgs[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            for tc in last.tool_calls:
                if tc.get("name") == "ask_user":
                    args = tc.get("args") or {}
                    return args.get("question")
    return None

def _inject_user_reply_as_tool_message(state_values: dict, user_reply: str):
    """Create a ToolMessage for the pending ask_user call."""
    for key in ("messages", "specialist_messages", "patho_messages", "radio_messages"):
        msgs: List = state_values.get(key) or []
        if not msgs:
            continue
        last = msgs[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            for tc in last.tool_calls:
                if tc.get("name") == "ask_user":
                    tool_call_id = tc.get("id")
                    return key, ToolMessage(content=user_reply, tool_call_id=tool_call_id)
    return None, None

def _last_assistant_text(state_values: dict) -> Optional[str]:
    """Get the text content of the last assistant message."""
    for key in ("specialist_messages", "patho_messages", "radio_messages", "messages"):
        msgs: List = state_values.get(key) or []
        if not msgs:
            continue
        last = msgs[-1]
        if hasattr(last, "content") and isinstance(last.content, str):
            text = last.content.strip()
            # Suppress bare routing tokens
            if key == "messages" and text.lower() in {
                "pediatrician", "pediatrics", "ophthalmologist", "orthopedist", "dermatologist",
                "ent", "gynecologist", "psychiatrist", "internal medicine"
            }:
                continue
            return text
    return None

def _speaker_for_key(stream_key: str) -> str:
    if stream_key == "messages":
        return "GP"
    if stream_key == "specialist_messages":
        return "Specialist"
    if stream_key == "patho_messages":
        return "Pathologist"
    if stream_key == "radio_messages":
        return "Radiologist"
    return "Assistant"

def _chunk_to_payload(chunk: dict) -> Optional[dict]:
    """Convert a graph chunk to a frontend payload."""
    current_agent = chunk.get("current_agent")
    for key in ("specialist_messages", "patho_messages", "radio_messages", "messages"):
        msgs: List = chunk.get(key) or []
        if not msgs:
            continue
        last = msgs[-1]
        if isinstance(last, AIMessage) and isinstance(last.content, str):
            text = last.content.strip()
            if key == "messages" and text.lower() in {
                "pediatrician", "pediatrics", "ophthalmologist", "orthopedist", "dermatologist",
                "ent", "gynecologist", "psychiatrist", "internal medicine"
            }:
                continue
            payload = {"content": text, "speaker": _speaker_for_key(key)}
            if current_agent:
                payload["current_agent"] = current_agent
            return payload
    if current_agent:
        return {"current_agent": current_agent}
    return None

def _new_tool_calls(chunk: dict, seen_ids: set) -> list[dict]:
    """Extract newly issued tool calls from the stream chunk."""
    out: list[dict] = []
    current_agent = chunk.get("current_agent")
    for key in ("specialist_messages", "patho_messages", "radio_messages", "messages"):
        msgs: List = chunk.get(key) or []
        if not msgs:
            continue
        last = msgs[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            agent_label = current_agent or _speaker_for_key(key)
            for tc in last.tool_calls:
                tc_id = tc.get("id")
                if not tc_id or tc_id in seen_ids:
                    continue
                seen_ids.add(tc_id)
                out.append({
                    "id": tc_id,
                    "name": tc.get("name"),
                    "args": tc.get("args", {}),
                    "agent": agent_label,
                })
    return out

# ---------------------------
# Endpoints
# ---------------------------
@router.get("/graph/start/stream")
async def start_graph_stream(
    message: str, 
    token: str, 
    db: Session = Depends(database.get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token_data = oauth2.verify_access_token(token, credentials_exception)
    patient_id = int(token_data.id)

    thread_id = str(uuid4())
    config = _make_config(thread_id)
    inputs = _initial_inputs(message, patient_id)

    async def event_gen():
        yield {"event": "thread", "data": json.dumps({"thread_id": thread_id})}
        current_agent = "GP"
        seen_tool_ids: set = set()
        
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
    token: str 
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
    )
    oauth2.verify_access_token(token, credentials_exception)
    
    config = _make_config(thread_id)
    state = myapp.get_state(config)
    if state is None:
        raise HTTPException(status_code=404, detail="Thread not found or expired")

    stream_key, tool_msg = _inject_user_reply_as_tool_message(state.values or {}, user_reply)
    if not tool_msg:
        raise HTTPException(status_code=400, detail="No pending ask_user call to answer")
    
    current_stream: List = state.values.get(stream_key, [])
    updated_stream = current_stream + [tool_msg]
    myapp.update_state(config, {stream_key: updated_stream})

    async def event_gen():
        current_agent = (state.values or {}).get("current_agent", "GP")
        seen_tool_ids: set = set()
        async for chunk in myapp.astream(None, config, stream_mode="values"):
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
        state2 = myapp.get_state(config)
        state_values2 = state2.values or {}
        next_nodes = set(state2.next or [])
        current_agent = state_values2.get("current_agent", current_agent)
        if next_nodes & ASK_NODES:
            question = _extract_ask_question(state_values2)
            ask_payload = {"thread_id": thread_id}
            if question:
                ask_payload["question"] = question
            if current_agent:
                ask_payload["current_agent"] = current_agent
                ask_payload["speaker"] = current_agent
            yield {"event": "ask_user", "data": json.dumps(ask_payload)}
        else:
            final = _last_assistant_text(state_values2)
            final_payload = {"thread_id": thread_id, "message": final}
            if current_agent:
                final_payload["current_agent"] = current_agent
            yield {"event": "final", "data": json.dumps(final_payload)}

    return EventSourceResponse(event_gen())