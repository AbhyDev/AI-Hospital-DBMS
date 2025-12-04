import os, sys
from typing import TypedDict, Annotated, List, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_tavily import TavilySearch
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from typing import Annotated, Sequence, TypedDict, List
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Knowledge_notebooks.initialize_rag import VectorRAG_initialize
from custom_libs.Audioconvert import text_to_speech, speech_to_text
vector_rag = VectorRAG_initialize()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    specialist_messages: Annotated[Sequence[BaseMessage], add_messages]
    patho_messages: Annotated[Sequence[BaseMessage], add_messages]
    radio_messages: Annotated[Sequence[BaseMessage], add_messages]
    radio_QnA: list[str]
    patho_QnA: list[str]
    next_agent: list[str]
    agent_order: list[str]
    current_report: list[str]
    current_agent: str
    consultation_id: Optional[int]
patient_info = ""
final_report = ""

from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted
from tenacity import wait_exponential # <-- Import this

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", # Using 1.5-flash as it's a common, recent model
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# llm = llm_base.with_retry(
#     stop_after_attempt=3,
#     # Correct way to set exponential backoff
#     wait=wait_exponential(multiplier=1, max=10), 
#     retry_if_exception_type=ResourceExhausted
# )

# Apply the same fix for the second instance
llm_rag = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# llm_rag = llm_rag_base.with_retry(
#     stop_after_attempt=3,
#     wait=wait_exponential(multiplier=1, max=10),
#     retry_if_exception_type=ResourceExhausted
# )

print("LLM instances with retry logic configured successfully.")

tavily_search = TavilySearch(
    max_results=5,
    search_depth="basic",
)

@tool
def ask_user(question: str) -> str:
    """Ask the user a question and get their response.

    Args:
        question (str): The question to ask the Patient.

    Returns:
        str: Answer provided by the Patient, or a status message if input fails.
    """
    # NOTE: In web apps we do not read from stdin here. This tool's call
    # is intercepted by the graph before execution (interrupt_before on
    # a dedicated *_AskUser node). The actual user answer is collected by
    # the frontend and injected on resume as a ToolMessage. Returning a
    # marker helps during local CLI runs.
    return f"ASK_USER:{question}"


@tool
def search_internet(query: str) -> str:
    """Search the internet for the given query and return the top results.

    Args:
        query (str): The search query.

    Returns:
        str: Top search results or an error message.
    """
    try:
        result = tavily_search.invoke({"query": query})
        # If result is structured (dict/list), format nicely
        if isinstance(result, (dict, list)):
            import json
            return json.dumps(result, indent=2)
        return str(result)
    except Exception as e:
        return f"Search failed: {str(e)}"


@tool
def add_report(report: str) -> str:
    """Add a report to the current patient's record.

    Args:
        report (str): The report content to add.

    Returns:
        str: Confirmation message.
    """
    return "Report added to patient's record."

@tool
def Patient_data_report(data: str)->str:
    """
    Process and store patient data for specialist recommendation.
    
    Args:
        data (str): Raw patient data including symptoms, history, and test results.
    
    Returns:
        str: Status message indicating data compiled and specialist recommendation needed.
    """
    global patient_info
    patient_info = data
    return "Patient Data compiled, Recommend a Specialist"

@tool
def VectorRAG_Retrival(query:str, agent:str)->str:
    """Retrieve and synthesize information from a domain-specific vector store.

    Args:
        query (str): The user's question to be answered.
        agent (str): The medical specialist domain from list: ['Ophthalmologist', 'Dermatology', 'ENT', 'Gynecology', 'Internal Medicine', 'Orthopedics', 'Pathology', 'Pediatrics', 'Psychiatry'], 
            (e.g., "Ophthalmologist") 
            used to select the appropriate vector store.

    Returns:
        str: A synthesized, context-based answer generated by the language model.

    """
    if "opthal" in agent.lower():
        agent = "Ophthalmologist"
    elif "derma" in agent.lower():
        agent = "Dermatology"
    elif "ent" in agent.lower():
        agent = "ENT"
    elif "gynec" in agent.lower():
        agent = "Gynecology"
    elif "internal" in agent.lower():
        agent = "Internal Medicine"
    elif "ortho" in agent.lower():
        agent = "Orthopedics"
    elif "patho" in agent.lower():
        agent = "Pathology"
    elif "pedia" in agent.lower():
        agent = "Pediatrics"
    elif "psych" in agent.lower():
        agent = "Psychiatry"

    retriever = vector_rag.vector_store[agent].as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.invoke(query)

    Systemprompt = SystemMessage(content=f"""
        response = pathllm.invoke([SystemPrompt] + state['patho_messages'])
        return {'patho_messages': [response], 'current_agent': 'Pathologist'}
    <context>
    {relevant_docs}
    </context>

    Based ONLY on the documents in the context above, provide a clear, consolidated answer to the following question. Use bullet points and headings if it improves clarity.

    Question: {query}

    If the documents do not contain enough information to form a comprehensive answer, you must state that a complete answer is not available in the provided text.
    """
    )
    response = llm_rag.invoke([Systemprompt]+[HumanMessage(content="Help me with this")])
    return response.content

gp_llm = llm.bind_tools([ask_user, Patient_data_report])
pediallm = llm.bind_tools([ask_user, search_internet, add_report, VectorRAG_Retrival])
ophthalllm = llm.bind_tools([ask_user, search_internet, add_report, VectorRAG_Retrival])
orthollm = llm.bind_tools([ask_user, search_internet, add_report, VectorRAG_Retrival])
dermallm = llm.bind_tools([ask_user, search_internet, add_report, VectorRAG_Retrival])
entllm = llm.bind_tools([ask_user, search_internet, add_report, VectorRAG_Retrival])
gynecllm = llm.bind_tools([ask_user, search_internet, add_report, VectorRAG_Retrival])
psychllm = llm.bind_tools([ask_user, search_internet, add_report, VectorRAG_Retrival])
intmedllm = llm.bind_tools([ask_user, search_internet, add_report, VectorRAG_Retrival])
radllm = llm.bind_tools([ask_user, search_internet, add_report])
pathllm = llm.bind_tools([ask_user, search_internet, add_report, VectorRAG_Retrival])

def general_physician(state: AgentState) -> AgentState:
    # Implement the logic for the general physician agent
    SystemPrompt = SystemMessage(content=f"""
You are a Medical Router AI / General Physician.

Your job is to assign patients to the MOST APPROPRIATE specialist from the list:
agents = ["Pediatrics", "Ophthalmology", "Orthopedist",
          "Dermatology", "ENT", "Gynecology", "Psychiatry",
          "Internal Medicine"]

Rules & Process:

1. You MUST ALWAYS use the ask_user tool to obtain ANY patient information (never ask questions directly in plain text). Ask ONE question per tool call.
   - Collect details on: symptoms, duration, location, severity, age, name, relevant history, child context.
   - If patient is a child, explicitly collect age, weight, and height.

2. Only AFTER you have enough information, call Patient_data_report with a concise structured summary (demographics + key symptoms + relevant negatives).

3. Just AFTER Patient_data_report has been called and confirmed (Patient_data_report status: {bool(patient_info)} is True), output EXACTLY the specialist name; return ONLY the name, nothing else.

4. Specialist criteria:

   - Pediatrics: Child/adolescent with general pediatric illnesses.
   - Ophthalmology: Eye problems (vision changes, redness, pain, floaters).
   - Orthopedics: Bone, joint, ligament, fracture, or chronic musculoskeletal pain.
   - Dermatology: Skin rashes, lesions, acne, eczema, unusual pigmentation.
   - ENT: Ear, nose, throat problems; hearing issues, sinusitis, sore throat.
   - Gynecology: Female reproductive complaints; menstrual, pregnancy, hormonal issues.
   - Psychiatry: Mental health, mood disorders, anxiety, depression, behavioral changes.
   - Internal Medicine: Adult patients with complex, chronic, multi-system diseases or systemic/unclear symptoms. Handles chronic disease management and coordinates care.

5. Helpers: Each specialist always has two helpers — Pathologist and Radiologist — who are automatically available. You do NOT assign them manually.

6. Triage Principles:
   - Ask clarifying questions ONE AT A TIME, until confident in specialist selection.
   - If symptoms overlap multiple specialties, prioritize the **underlying cause** over just local symptoms.
   - Do NOT prescribe medications; your role is purely triage.
   - Never guess age or other demographic details — always collect via ask_user.
7. If you do not call any tool and neither a valid specialist name, it will simply loop back to you, try avoiding that.
Begin by greeting the patient, then ask the first clarifying question using ask_user.
""")

    response = gp_llm.invoke([SystemPrompt]+state['messages'])
    return {'messages' : [response], 'current_agent': 'GP'}




def router_gp(state: AgentState) -> AgentState:
    last_message = state['messages'][-1]
    content = last_message.content.lower()
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        # Route ask_user separately so we can interrupt before it
        ask_user_called = any(tc.get('name') == 'ask_user' for tc in last_message.tool_calls)
        if ask_user_called:
            return "GP_AskUser"
        return "GP_Tooler"
    elif "pediatrics" in content or "pediatrician" in content:
        state['next_agent'].append('Pediatrician')
        return "pediatrics"
    elif "ophthalmology" in content or "ophthalmologist" in content:
        state['next_agent'].append('Ophthalmologist')
        return "ophthalmology"
    elif "Orthopedist" in content or "orthopedist" in content:
        state['next_agent'].append('Orthopedist')
        return "Orthopedics"
    elif "dermatology" in content or "dermatologist" in content:
        state['next_agent'].append('Dermatologist')
        return "dermatology"
    elif "gynecology" in content or "gynecologist" in content:
        state['next_agent'].append('Gynecologist')
        return "gynecology"
    elif "psychiatry" in content or "psychiatrist" in content:
        state['next_agent'].append('Psychiatrist')
        return "psychiatry"
    elif "internal medicine" in content or "internal" in content:
        state['next_agent'].append('Internal Medicine')
        return "internal medicine"
    elif "ent" in content:
        state['next_agent'].append('ENT')
        return "ent"
    else:
        return "GP"


def Ophthalmologist(state: AgentState) -> AgentState:
    # Implement the logic for the ophthalmologist agent
    global patient_info
    SystemPrompt = SystemMessage(content=f"""You are a High Quality Ophthalmologist.

Your patient's initial data is: {patient_info}.

Current status:
1. Status of Radiologist QnA: {", ".join(state['radio_QnA']) if state['radio_QnA'] else "None"}.
2. Status of Pathologist QnA: {", ".join(state['patho_QnA']) if state['patho_QnA'] else "None"}.
3. Current report status: {", ".join(state['current_report']) if state['current_report'] else "None"}.

You have access to three tools:
1. **ask_user** - Use this tool to ask the patient any questions you need answered. 
2. **search_internet** - Use this tool to look up any medical information you need. 
3. **add_report** - Use this tool to add relevant findings to the report. You can call it multiple times. 'current_report' will include Pathologist and Radiologist findings after you request their help.
4. **VectorRAG_Retrival(query:str, agent:str)** - Use this tool to retrieve and synthesize knowledge from a high-quality vector store of medical books and guidelines.  
   - Always pass the correct `agent` domain (e.g., "Ophthalmologist").  
   - You may use it any number of times whenever deeper, authoritative medical knowledge is needed.  
   - If the first query does not provide a satisfactory answer, you may try **one or two re-phrased queries**, but do not enter an infinite loop.

Your tasks:
1. **Ask Questions**: If more patient information is needed, you MUST use the 'ask_user' tool. Ask one question at a time.
2. **Use Helpers**: If you need a Pathologist or Radiologist, output plain text like: 
   - "I need a blood report from Pathologist, (your question)"
   - "I need imaging studies from Radiologist, (your question)"
3. **Use Knowledge Bases**: If medical domain expertise is required, you may use the VectorRAG_Retrival tool. Prefer it over raw internet search for authoritative textbook knowledge.
4. **Final Analysis & Reporting (VERY LAST ACTION):**
   Only when you have gathered ALL necessary information (from the patient, helpers, and internet searches) should you begin the final two-step reporting process.

   **Step A (Log Final Summary):** First, you MUST call the `add_report` tool one last time. This single tool call must contain your complete, synthesized findings, including a definitive Diagnosis, a full Treatment Plan, and clear Follow-up Instructions.

   **Step B (Output Final Report):** After the final `add_report` tool call is confirmed, you MUST immediately output the complete, human-readable report using the exact phrase `Final Report: (your full report text)`. Do not perform any other actions after this.
Rules:
- Always use the 'ask_user' tool for questions to the patient.
- The conversation continues until a Final Report is produced.
- Responses from Pathologist or Radiologist will automatically be added to your context.
- Never ask multiple questions in one tool call.
- Do not loop indefinitely with VectorRAG_Retrival: maximum 2 reformulations if the first query fails.
- If you return plain text that does not mention 'pathologist', 'radiologist', or 'Final Report:', it will be ignored.

""")

    
    response = ophthalllm.invoke([SystemPrompt]+state['specialist_messages']) 
    return {'specialist_messages' : [response], 'current_agent': 'Ophthalmologist'}

def router_opthal(state: AgentState) -> AgentState:
    # Route the request to the appropriate ophthalmologist agent
    global final_report
    last_message = state['specialist_messages'][-1]
    content = last_message.content.lower()

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        ask_user_called = any(tc.get('name') == 'ask_user' for tc in last_message.tool_calls)
        for tool_call in last_message.tool_calls:
            if tool_call['name'] == 'add_report':
                report_content = tool_call['args']['report']
                state['current_report'].append(report_content)

        if ask_user_called:
            return "Ophthal_AskUser"
        return "Ophthal_Tooler"
    
    elif "pathologist" in content:
        # Record caller so helpers know who invoked them
        state['next_agent'].append('Ophthalmologist')
        state["patho_QnA"].append("Question from Ophthalmologist to Pathologist: ")
        state["patho_QnA"].append(last_message.content)
        return "Pathologist"
    elif "radiologist" in content:
        # Record caller so helpers know who invoked them
        state['next_agent'].append('Ophthalmologist')
        state['radio_QnA'].append("Question from Ophthalmologist to Radiologist: ")
        state['radio_QnA'].append(last_message.content)
        return "Radiologist"
    elif "final report:" in content:
        global final_report 
        final_report = state['current_report']
        return "end"
    else:
        return "Ophthalmologist"

def Pediatrician(state: AgentState) -> AgentState:
    # Implement the logic for the pediatrician agent
    global patient_info
    SystemPrompt = SystemMessage(content=f"""You are a High Quality Pediatrician.

Your patient's initial data is: {patient_info}.

Current status:
1. Status of Radiologist QnA: {", ".join(state['radio_QnA']) if state['radio_QnA'] else "None"}.
2. Status of Pathologist QnA: {", ".join(state['patho_QnA']) if state['patho_QnA'] else "None"}.
3. Current report status: {", ".join(state['current_report']) if state['current_report'] else "None"}.

You have access to three tools:
1. **ask_user** - Use this tool to ask the patient any questions you need answered. 
2. **search_internet** - Use this tool to look up any medical information you need. 
3. **add_report** - Use this tool to add relevant findings to the report. You can call it multiple times. 'current_report' will include Pathologist and Radiologist findings after you request their help.
4. **VectorRAG_Retrival(query:str, agent:str)** - Use this tool to retrieve and synthesize knowledge from a high-quality vector store of medical books and guidelines.  
   - Always pass the correct `agent` domain ("Pediatrics").  
   - You may use it any number of times whenever deeper, authoritative medical knowledge is needed.  
   - If the first query does not provide a satisfactory answer, you may try **one or two re-phrased queries**, but do not enter an infinite loop.

Your tasks:
1. **Ask Questions**: If more patient information is needed, you MUST use the 'ask_user' tool. Ask one question at a time.
2. **Use Helpers**: If you need a Pathologist or Radiologist, output plain text like: 
   - "I need a blood report from Pathologist, (your question)"
   - "I need imaging studies from Radiologist, (your question)"
3. **Use Knowledge Bases**: If medical domain expertise is required, you may use the VectorRAG_Retrival tool. Prefer it over raw internet search for authoritative textbook knowledge.
4. **Final Analysis & Reporting (VERY LAST ACTION):**
   Only when you have gathered ALL necessary information (from the patient, helpers, and internet searches) should you begin the final two-step reporting process.

   **Step A (Log Final Summary):** First, you MUST call the `add_report` tool one last time. This single tool call must contain your complete, synthesized findings, including a definitive Diagnosis, a full Treatment Plan, and clear Follow-up Instructions.

   **Step B (Output Final Report):** After the final `add_report` tool call is confirmed, you MUST immediately output the complete, human-readable report using the exact phrase `Final Report: (your full report text)`. Do not perform any other actions after this.
Rules:
- Always use the 'ask_user' tool for questions to the patient.
- The conversation continues until a Final Report is produced.
- Responses from Pathologist or Radiologist will automatically be added to your context.
- Never ask multiple questions in one tool call.
- Do not loop indefinitely with VectorRAG_Retrival: maximum 2 reformulations if the first query fails.
- If you return plain text that does not mention 'pathologist', 'radiologist', or 'Final Report:', it will be ignored.

""")

    
    response = pediallm.invoke([SystemPrompt]+state['specialist_messages']) 
    return {'specialist_messages' : [response], 'current_agent': 'Pediatrician'}

def router_pedia(state: AgentState) -> AgentState:
    # Route the request to the appropriate ophthalmologist agent
    global final_report
    last_message = state['specialist_messages'][-1]
    content = last_message.content.lower()

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        ask_user_called = any(tc.get('name') == 'ask_user' for tc in last_message.tool_calls)
        for tool_call in last_message.tool_calls:
            if tool_call['name'] == 'add_report':
                report_content = tool_call['args']['report']
                state['current_report'].append(report_content)
        if ask_user_called:
            return "Pedia_AskUser"
        return "Pedia_Tooler"
    
    elif "pathologist" in content:
        # Record caller so helpers know who invoked them
        state['next_agent'].append('Pediatrician')
        state["patho_QnA"].append("Question from Pediatrician to Pathologist: ")
        state["patho_QnA"].append(last_message.content)
        return "Pathologist"
    elif "radiologist" in content:
        # Record caller so helpers know who invoked them
        state['next_agent'].append('Pediatrician')
        state['radio_QnA'].append("Question from Pediatrician to Radiologist: ")
        state['radio_QnA'].append(last_message.content)
        return "Radiologist"
    elif "final report:" in content:
        global final_report 
        final_report = state['current_report']
        return "end"
    else:
        return "Pediatrician"

def Orthopedist(state: AgentState) -> AgentState:
    # Implement the logic for the orthopedist agent
    global patient_info
    SystemPrompt = SystemMessage(content=f"""You are a High Quality Orthopedist.

Your patient's initial data is: {patient_info}.

Current status:
1. Status of Radiologist QnA: {", ".join(state['radio_QnA']) if state['radio_QnA'] else "None"}.
2. Status of Pathologist QnA: {", ".join(state['patho_QnA']) if state['patho_QnA'] else "None"}.
3. Current report status: {", ".join(state['current_report']) if state['current_report'] else "None"}.

You have access to three tools:
1. **ask_user** - Use this tool to ask the patient any questions you need answered. 
2. **search_internet** - Use this tool to look up any medical information you need. 
3. **add_report** - Use this tool to add relevant findings to the report. You can call it multiple times. 'current_report' will include Pathologist and Radiologist findings after you request their help.
4. **VectorRAG_Retrival(query:str, agent:str)** - Use this tool to retrieve and synthesize knowledge from a high-quality vector store of medical books and guidelines.  
   - Always pass the correct `agent` domain (Orthopedics).  
   - You may use it any number of times whenever deeper, authoritative medical knowledge is needed.  
   - If the first query does not provide a satisfactory answer, you may try **one or two re-phrased queries**, but do not enter an infinite loop.

Your tasks:
1. **Ask Questions**: If more patient information is needed, you MUST use the 'ask_user' tool. Ask one question at a time.
2. **Use Helpers**: If you need a Pathologist or Radiologist, output plain text like: 
   - "I need a blood report from Pathologist, (your question)"
   - "I need imaging studies from Radiologist, (your question)"
3. **Use Knowledge Bases**: If medical domain expertise is required, you may use the VectorRAG_Retrival tool. Prefer it over raw internet search for authoritative textbook knowledge.
4. **Final Analysis & Reporting (VERY LAST ACTION):**
   Only when you have gathered ALL necessary information (from the patient, helpers, and internet searches) should you begin the final two-step reporting process.

   **Step A (Log Final Summary):** First, you MUST call the `add_report` tool one last time. This single tool call must contain your complete, synthesized findings, including a definitive Diagnosis, a full Treatment Plan, and clear Follow-up Instructions.

   **Step B (Output Final Report):** After the final `add_report` tool call is confirmed, you MUST immediately output the complete, human-readable report using the exact phrase `Final Report: (your full report text)`. Do not perform any other actions after this.
Rules:
- Always use the 'ask_user' tool for questions to the patient.
- The conversation continues until a Final Report is produced.
- Responses from Pathologist or Radiologist will automatically be added to your context.
- Never ask multiple questions in one tool call.
- Do not loop indefinitely with VectorRAG_Retrival: maximum 2 reformulations if the first query fails.
- If you return plain text that does not mention 'pathologist', 'radiologist', or 'Final Report:', it will be ignored.

""")

    
    response = orthollm.invoke([SystemPrompt]+state['specialist_messages']) 
    return {'specialist_messages' : [response], 'current_agent': 'Orthopedist'}

def router_ortho(state: AgentState) -> AgentState:
    # Route the request to the appropriate ophthalmologist agent
    global final_report
    last_message = state['specialist_messages'][-1]
    content = last_message.content.lower()

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        ask_user_called = any(tc.get('name') == 'ask_user' for tc in last_message.tool_calls)
        for tool_call in last_message.tool_calls:
            if tool_call['name'] == 'add_report':
                report_content = tool_call['args']['report']
                state['current_report'].append(report_content)
        if ask_user_called:
            return "Ortho_AskUser"
        return "Ortho_Tooler"
    
    elif "pathologist" in content:
        # Record caller so helpers know who invoked them
        state['next_agent'].append('Orthopedist')
        state["patho_QnA"].append("Question from Orthopedist to Pathologist: ")
        state["patho_QnA"].append(last_message.content)
        return "Pathologist"
    elif "radiologist" in content:
        # Record caller so helpers know who invoked them
        state['next_agent'].append('Orthopedist')
        state['radio_QnA'].append("Question from Orthopedist to Radiologist: ")
        state['radio_QnA'].append(last_message.content)
        return "Radiologist"
    elif "final report:" in content:
        global final_report 
        final_report = state['current_report']
        return "end"
    else:
        return "Orthopedist"

def Dermatologist(state: AgentState) -> AgentState:
    # Implement the logic for the dermatologist agent
    global patient_info
    SystemPrompt = SystemMessage(content=f"""You are a High Quality Dermatologist.

Your patient's initial data is: {patient_info}.

Current status:
1. Status of Radiologist QnA: {", ".join(state['radio_QnA']) if state['radio_QnA'] else "None"}.
2. Status of Pathologist QnA: {", ".join(state['patho_QnA']) if state['patho_QnA'] else "None"}.
3. Current report status: {", ".join(state['current_report']) if state['current_report'] else "None"}.

You have access to three tools:
1. **ask_user** - Use this tool to ask the patient any questions you need answered. 
2. **search_internet** - Use this tool to look up any medical information you need. 
3. **add_report** - Use this tool to add relevant findings to the report. You can call it multiple times. 'current_report' will include Pathologist and Radiologist findings after you request their help.
4. **VectorRAG_Retrival(query:str, agent:str)** - Use this tool to retrieve and synthesize knowledge from a high-quality vector store of medical books and guidelines.  
   - Always pass the correct `agent` domain (Dermatology).  
   - You may use it any number of times whenever deeper, authoritative medical knowledge is needed.  
   - If the first query does not provide a satisfactory answer, you may try **one or two re-phrased queries**, but do not enter an infinite loop.

Your tasks:
1. **Ask Questions**: If more patient information is needed, you MUST use the 'ask_user' tool. Ask one question at a time.
2. **Use Helpers**: If you need a Pathologist or Radiologist, output plain text like: 
   - "I need a blood report from Pathologist, (your question)"
   - "I need imaging studies from Radiologist, (your question)"
3. **Use Knowledge Bases**: If medical domain expertise is required, you may use the VectorRAG_Retrival tool. Prefer it over raw internet search for authoritative textbook knowledge.
4. **Final Analysis & Reporting (VERY LAST ACTION):**
   Only when you have gathered ALL necessary information (from the patient, helpers, and internet searches) should you begin the final two-step reporting process.

   **Step A (Log Final Summary):** First, you MUST call the `add_report` tool one last time. This single tool call must contain your complete, synthesized findings, including a definitive Diagnosis, a full Treatment Plan, and clear Follow-up Instructions.

   **Step B (Output Final Report):** After the final `add_report` tool call is confirmed, you MUST immediately output the complete, human-readable report using the exact phrase `Final Report: (your full report text)`. Do not perform any other actions after this.
Rules:
- Always use the 'ask_user' tool for questions to the patient.
- The conversation continues until a Final Report is produced.
- Responses from Pathologist or Radiologist will automatically be added to your context.
- Never ask multiple questions in one tool call.
- Do not loop indefinitely with VectorRAG_Retrival: maximum 2 reformulations if the first query fails.
- If you return plain text that does not mention 'pathologist', 'radiologist', or 'Final Report:', it will be ignored.

""")

    
    response = dermallm.invoke([SystemPrompt]+state['specialist_messages']) 
    return {'specialist_messages' : [response], 'current_agent': 'Dermatologist'}

def router_dermat(state: AgentState) -> AgentState:
    # Route the request to the appropriate dermatologist agent
    global final_report
    last_message = state['specialist_messages'][-1]
    content = last_message.content.lower()

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        ask_user_called = any(tc.get('name') == 'ask_user' for tc in last_message.tool_calls)
        for tool_call in last_message.tool_calls:
            if tool_call['name'] == 'add_report':
                report_content = tool_call['args']['report']
                state['current_report'].append(report_content)
        if ask_user_called:
            return "Dermat_AskUser"
        return "Dermat_Tooler"
    
    elif "pathologist" in content:
        # Record caller so helpers know who invoked them
        state['next_agent'].append('Dermatologist')
        state["patho_QnA"].append("Question from Dermatologist to Pathologist: ")
        state["patho_QnA"].append(last_message.content)
        return "Pathologist"
    elif "radiologist" in content:
        # Record caller so helpers know who invoked them
        state['next_agent'].append('Dermatologist')
        state['radio_QnA'].append("Question from Dermatologist to Radiologist: ")
        state['radio_QnA'].append(last_message.content)
        return "Radiologist"
    elif "final report:" in content:
        global final_report 
        final_report = state['current_report']
        return "end"
    else:
        return "Dermatologist"

def ENT(state: AgentState) -> AgentState:
    # Implement the logic for the ENT agent
    global patient_info
    SystemPrompt = SystemMessage(content=f"""You are a High Quality ENT Specialist.

Your patient's initial data is: {patient_info}.

Current status:
1. Status of Radiologist QnA: {", ".join(state['radio_QnA']) if state['radio_QnA'] else "None"}.
2. Status of Pathologist QnA: {", ".join(state['patho_QnA']) if state['patho_QnA'] else "None"}.
3. Current report status: {", ".join(state['current_report']) if state['current_report'] else "None"}.

You have access to three tools:
1. **ask_user** - Use this tool to ask the patient any questions you need answered. 
2. **search_internet** - Use this tool to look up any medical information you need. 
3. **add_report** - Use this tool to add relevant findings to the report. You can call it multiple times. 'current_report' will include Pathologist and Radiologist findings after you request their help.
4. **VectorRAG_Retrival(query:str, agent:str)** - Use this tool to retrieve and synthesize knowledge from a high-quality vector store of medical books and guidelines.  
   - Always pass the correct `agent` domain (ENT).  
   - You may use it any number of times whenever deeper, authoritative medical knowledge is needed.  
   - If the first query does not provide a satisfactory answer, you may try **one or two re-phrased queries**, but do not enter an infinite loop.

Your tasks:
1. **Ask Questions**: If more patient information is needed, you MUST use the 'ask_user' tool. Ask one question at a time.
2. **Use Helpers**: If you need a Pathologist or Radiologist, output plain text like: 
   - "I need a blood report from Pathologist, (your question)"
   - "I need imaging studies from Radiologist, (your question)"
3. **Use Knowledge Bases**: If medical domain expertise is required, you may use the VectorRAG_Retrival tool. Prefer it over raw internet search for authoritative textbook knowledge.
4. **Final Analysis & Reporting (VERY LAST ACTION):**
   Only when you have gathered ALL necessary information (from the patient, helpers, and internet searches) should you begin the final two-step reporting process.

   **Step A (Log Final Summary):** First, you MUST call the `add_report` tool one last time. This single tool call must contain your complete, synthesized findings, including a definitive Diagnosis, a full Treatment Plan, and clear Follow-up Instructions.

   **Step B (Output Final Report):** After the final `add_report` tool call is confirmed, you MUST immediately output the complete, human-readable report using the exact phrase `Final Report: (your full report text)`. Do not perform any other actions after this.
Rules:
- Always use the 'ask_user' tool for questions to the patient.
- The conversation continues until a Final Report is produced.
- Responses from Pathologist or Radiologist will automatically be added to your context.
- Never ask multiple questions in one tool call.
- Do not loop indefinitely with VectorRAG_Retrival: maximum 2 reformulations if the first query fails.
- If you return plain text that does not mention 'pathologist', 'radiologist', or 'Final Report:', it will be ignored.

""")

    
    response = entllm.invoke([SystemPrompt]+state['specialist_messages']) 
    return {'specialist_messages' : [response], 'current_agent': 'ENT'}

def router_ent(state: AgentState) -> AgentState:
    # Route the request to the appropriate ENT agent
    global final_report
    last_message = state['specialist_messages'][-1]
    content = last_message.content.lower()

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        ask_user_called = any(tc.get('name') == 'ask_user' for tc in last_message.tool_calls)
        for tool_call in last_message.tool_calls:
            if tool_call['name'] == 'add_report':
                report_content = tool_call['args']['report']
                state['current_report'].append(report_content)
        if ask_user_called:
            return "ENT_AskUser"
        return "ENT_Tooler"

    elif "pathologist" in content:
        # Record caller so helpers know who invoked them
        state['next_agent'].append('ENT')
        state["patho_QnA"].append("Question from ENT to Pathologist: ")
        state["patho_QnA"].append(last_message.content)
        return "Pathologist"
    elif "radiologist" in content:
        # Record caller so helpers know who invoked them
        state['next_agent'].append('ENT')
        state['radio_QnA'].append("Question from ENT to Radiologist: ")
        state['radio_QnA'].append(last_message.content)
        return "Radiologist"
    elif "final report:" in content:
        global final_report 
        final_report = state['current_report']
        return "end"
    else:
        return "ENT"

def Gynecologist(state: AgentState) -> AgentState:
    # Implement the logic for the Gynecologist agent
    global patient_info
    SystemPrompt = SystemMessage(content=f"""You are a High Quality Gynecologist.

Your patient's initial data is: {patient_info}.

Current status:
1. Status of Radiologist QnA: {", ".join(state['radio_QnA']) if state['radio_QnA'] else "None"}.
2. Status of Pathologist QnA: {", ".join(state['patho_QnA']) if state['patho_QnA'] else "None"}.
3. Current report status: {", ".join(state['current_report']) if state['current_report'] else "None"}.

You have access to three tools:
1. **ask_user** - Use this tool to ask the patient any questions you need answered. 
2. **search_internet** - Use this tool to look up any medical information you need. 
3. **add_report** - Use this tool to add relevant findings to the report. You can call it multiple times. 'current_report' will include Pathologist and Radiologist findings after you request their help.
4. **VectorRAG_Retrival(query:str, agent:str)** - Use this tool to retrieve and synthesize knowledge from a high-quality vector store of medical books and guidelines.  
   - Always pass the correct `agent` domain (Gynecology).  
   - You may use it any number of times whenever deeper, authoritative medical knowledge is needed.  
   - If the first query does not provide a satisfactory answer, you may try **one or two re-phrased queries**, but do not enter an infinite loop.

Your tasks:
1. **Ask Questions**: If more patient information is needed, you MUST use the 'ask_user' tool. Ask one question at a time.
2. **Use Helpers**: If you need a Pathologist or Radiologist, output plain text like: 
   - "I need a blood report from Pathologist, (your question)"
   - "I need imaging studies from Radiologist, (your question)"
3. **Use Knowledge Bases**: If medical domain expertise is required, you may use the VectorRAG_Retrival tool. Prefer it over raw internet search for authoritative textbook knowledge.
4. **Final Analysis & Reporting (VERY LAST ACTION):**
   Only when you have gathered ALL necessary information (from the patient, helpers, and internet searches) should you begin the final two-step reporting process.

   **Step A (Log Final Summary):** First, you MUST call the `add_report` tool one last time. This single tool call must contain your complete, synthesized findings, including a definitive Diagnosis, a full Treatment Plan, and clear Follow-up Instructions.

   **Step B (Output Final Report):** After the final `add_report` tool call is confirmed, you MUST immediately output the complete, human-readable report using the exact phrase `Final Report: (your full report text)`. Do not perform any other actions after this.
Rules:
- Always use the 'ask_user' tool for questions to the patient.
- The conversation continues until a Final Report is produced.
- Responses from Pathologist or Radiologist will automatically be added to your context.
- Never ask multiple questions in one tool call.
- Do not loop indefinitely with VectorRAG_Retrival: maximum 2 reformulations if the first query fails.
- If you return plain text that does not mention 'pathologist', 'radiologist', or 'Final Report:', it will be ignored.

""")

    
    response = gynecllm.invoke([SystemPrompt]+state['specialist_messages']) 
    return {'specialist_messages' : [response], 'current_agent': 'Gynecologist'}

def router_gynec(state: AgentState) -> AgentState:
    # Route the request to the appropriate gynecologist agent
    global final_report
    last_message = state['specialist_messages'][-1]
    content = last_message.content.lower()

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        ask_user_called = any(tc.get('name') == 'ask_user' for tc in last_message.tool_calls)
        for tool_call in last_message.tool_calls:
            if tool_call['name'] == 'add_report':
                report_content = tool_call['args']['report']
                state['current_report'].append(report_content)
        if ask_user_called:
            return "Gynec_AskUser"
        return "Gynec_Tooler"
    
    elif "pathologist" in content:
        # Record caller so helpers know who invoked them
        state['next_agent'].append('Gynecologist')
        state["patho_QnA"].append("Question from Gynecologist to Pathologist: ")
        state["patho_QnA"].append(last_message.content)
        return "Pathologist"
    elif "radiologist" in content:
        # Record caller so helpers know who invoked them
        state['next_agent'].append('Gynecologist')
        state['radio_QnA'].append("Question from Gynecologist to Radiologist: ")
        state['radio_QnA'].append(last_message.content)
        return "Radiologist"
    elif "final report:" in content:
        global final_report 
        final_report = state['current_report']
        return "end"
    else:
        return "Gynecologist"

def Psychiatrist(state: AgentState) -> AgentState:
    # Implement the logic for the psychiatrist agent
    global patient_info
    SystemPrompt = SystemMessage(content=f"""You are a High Quality Psychiatrist.

Your patient's initial data is: {patient_info}.

Current status:
1. Status of Radiologist QnA: {", ".join(state['radio_QnA']) if state['radio_QnA'] else "None"}.
2. Status of Pathologist QnA: {", ".join(state['patho_QnA']) if state['patho_QnA'] else "None"}.
3. Current report status: {", ".join(state['current_report']) if state['current_report'] else "None"}.

You have access to three tools:
1. **ask_user** - Use this tool to ask the patient any questions you need answered. 
2. **search_internet** - Use this tool to look up any medical information you need. 
3. **add_report** - Use this tool to add relevant findings to the report. You can call it multiple times. 'current_report' will include Pathologist and Radiologist findings after you request their help.
4. **VectorRAG_Retrival(query:str, agent:str)** - Use this tool to retrieve and synthesize knowledge from a high-quality vector store of medical books and guidelines.  
   - Always pass the correct `agent` domain (Psychiatry).  
   - You may use it any number of times whenever deeper, authoritative medical knowledge is needed.  
   - If the first query does not provide a satisfactory answer, you may try **one or two re-phrased queries**, but do not enter an infinite loop.

Your tasks:
1. **Ask Questions**: If more patient information is needed, you MUST use the 'ask_user' tool. Ask one question at a time.
2. **Use Helpers**: If you need a Pathologist or Radiologist, output plain text like: 
   - "I need a blood report from Pathologist, (your question)"
   - "I need imaging studies from Radiologist, (your question)"
3. **Use Knowledge Bases**: If medical domain expertise is required, you may use the VectorRAG_Retrival tool. Prefer it over raw internet search for authoritative textbook knowledge.
4. **Final Analysis & Reporting (VERY LAST ACTION):**
   Only when you have gathered ALL necessary information (from the patient, helpers, and internet searches) should you begin the final two-step reporting process.

   **Step A (Log Final Summary):** First, you MUST call the `add_report` tool one last time. This single tool call must contain your complete, synthesized findings, including a definitive Diagnosis, a full Treatment Plan, and clear Follow-up Instructions.

   **Step B (Output Final Report):** After the final `add_report` tool call is confirmed, you MUST immediately output the complete, human-readable report using the exact phrase `Final Report: (your full report text)`. Do not perform any other actions after this.
Rules:
- Always use the 'ask_user' tool for questions to the patient.
- The conversation continues until a Final Report is produced.
- Responses from Pathologist or Radiologist will automatically be added to your context.
- Never ask multiple questions in one tool call.
- Do not loop indefinitely with VectorRAG_Retrival: maximum 2 reformulations if the first query fails.
- If you return plain text that does not mention 'pathologist', 'radiologist', or 'Final Report:', it will be ignored.

""")

    
    response = psychllm.invoke([SystemPrompt]+state['specialist_messages']) 
    return {'specialist_messages' : [response], 'current_agent': 'Psychiatrist'}

def router_psych(state: AgentState) -> AgentState:
    # Route the request to the appropriate psychiatrist agent
    global final_report
    last_message = state['specialist_messages'][-1]
    content = last_message.content.lower()

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        ask_user_called = any(tc.get('name') == 'ask_user' for tc in last_message.tool_calls)
        for tool_call in last_message.tool_calls:
            if tool_call['name'] == 'add_report':
                report_content = tool_call['args']['report']
                state['current_report'].append(report_content)
        if ask_user_called:
            return "Psych_AskUser"
        return "Psych_Tooler"

    elif "pathologist" in content:
        # Record caller so helpers know who invoked them
        state['next_agent'].append('Psychiatrist')
        state["patho_QnA"].append("Question from Psychiatrist to Pathologist: ")
        state["patho_QnA"].append(last_message.content)
        return "Pathologist"
    elif "radiologist" in content:
        # Record caller so helpers know who invoked them
        state['next_agent'].append('Psychiatrist')
        state['radio_QnA'].append("Question from Psychiatrist to Radiologist: ")
        state['radio_QnA'].append(last_message.content)
        return "Radiologist"
    elif "final report:" in content:
        global final_report 
        final_report = state['current_report']
        return "end"
    else:
        return "Psychiatrist"


def Internal_Medicine(state: AgentState) -> AgentState:
    # Implement the logic for the Internal Medicine agent
    global patient_info
    SystemPrompt = SystemMessage(content=f"""You are a High Quality Internal Medicine Specialist.

Your patient's initial data is: {patient_info}.

Current status:
1. Status of Radiologist QnA: {", ".join(state['radio_QnA']) if state['radio_QnA'] else "None"}.
2. Status of Pathologist QnA: {", ".join(state['patho_QnA']) if state['patho_QnA'] else "None"}.
3. Current report status: {", ".join(state['current_report']) if state['current_report'] else "None"}.

You have access to three tools:
1. **ask_user** - Use this tool to ask the patient any questions you need answered. 
2. **search_internet** - Use this tool to look up any medical information you need. 
3. **add_report** - Use this tool to add relevant findings to the report. You can call it multiple times. 'current_report' will include Pathologist and Radiologist findings after you request their help.
4. **VectorRAG_Retrival(query:str, agent:str)** - Use this tool to retrieve and synthesize knowledge from a high-quality vector store of medical books and guidelines.  
   - Always pass the correct `agent` domain (Internal Medicine).  
   - You may use it any number of times whenever deeper, authoritative medical knowledge is needed.  
   - If the first query does not provide a satisfactory answer, you may try **one or two re-phrased queries**, but do not enter an infinite loop.

Your tasks:
1. **Ask Questions**: If more patient information is needed, you MUST use the 'ask_user' tool. Ask one question at a time.
2. **Use Helpers**: If you need a Pathologist or Radiologist, output plain text like: 
   - "I need a blood report from Pathologist, (your question)"
   - "I need imaging studies from Radiologist, (your question)"
3. **Use Knowledge Bases**: If medical domain expertise is required, you may use the VectorRAG_Retrival tool. Prefer it over raw internet search for authoritative textbook knowledge.
4. **Final Analysis & Reporting (VERY LAST ACTION):**
   Only when you have gathered ALL necessary information (from the patient, helpers, and internet searches) should you begin the final two-step reporting process.

   **Step A (Log Final Summary):** First, you MUST call the `add_report` tool one last time. This single tool call must contain your complete, synthesized findings, including a definitive Diagnosis, a full Treatment Plan, and clear Follow-up Instructions.

   **Step B (Output Final Report):** After the final `add_report` tool call is confirmed, you MUST immediately output the complete, human-readable report using the exact phrase `Final Report: (your full report text)`. Do not perform any other actions after this.
Rules:
- Always use the 'ask_user' tool for questions to the patient.
- The conversation continues until a Final Report is produced.
- Responses from Pathologist or Radiologist will automatically be added to your context.
- Never ask multiple questions in one tool call.
- Do not loop indefinitely with VectorRAG_Retrival: maximum 2 reformulations if the first query fails.
- If you return plain text that does not mention 'pathologist', 'radiologist', or 'Final Report:', it will be ignored.

""")

    
    response = intmedllm.invoke([SystemPrompt]+state['specialist_messages']) 
    return {'specialist_messages' : [response], 'current_agent': 'Internal Medicine'}

def router_medicine(state: AgentState) -> AgentState:
    # Route the request to the appropriate internal medicine agent
    global final_report
    last_message = state['specialist_messages'][-1]
    content = last_message.content.lower()

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        ask_user_called = any(tc.get('name') == 'ask_user' for tc in last_message.tool_calls)
        for tool_call in last_message.tool_calls:
            if tool_call['name'] == 'add_report':
                report_content = tool_call['args']['report']
                state['current_report'].append(report_content)
        if ask_user_called:
            return "IntMed_AskUser"
        return "IntMed_Tooler"
    
    elif "pathologist" in content:
        # Record caller so helpers know who invoked them
        state['next_agent'].append('Internal Medicine')
        state["patho_QnA"].append("Question from Internal Medicine to Pathologist: ")
        state["patho_QnA"].append(last_message.content)
        return "Pathologist"
    elif "radiologist" in content:
        # Record caller so helpers know who invoked them
        state['next_agent'].append('Internal Medicine')
        state['radio_QnA'].append("Question from Internal Medicine to Radiologist: ")
        state['radio_QnA'].append(last_message.content)
        return "Radiologist"
    elif "final report:" in content:
        global final_report 
        final_report = state['current_report']
        return "end"
    else:
        return "Internal Medicine"

def Pathologist(state: AgentState) -> AgentState:
    # Implement the logic for the pathologist agent
    global patient_info
    callers = state.get('next_agent') or []
    caller = callers[-1] if callers else "General Physician"
    SystemPrompt = SystemMessage(content=f"""You are a High Quality Pathologist. You were called in by specialist {caller}.

    Patient data: {patient_info}

    Current context:
    - Total conversation with specialist: {", ".join(state['patho_QnA']) if state['patho_QnA'] else "None"}
    - If the last message is a question from the specialist, frame your response accordingly. You may also use tools if needed.

    Tools available:
    1. **ask_user** - Use this tool to ask the patient any details that might help your analysis. You may call it multiple times. 
    2. **search_internet** - Use this tool to search for any medical query at any point.
    3. **add_report** - Use this tool to add all final findings to the report. Always call it before returning your final plain text summary.
    4. **VectorRAG_Retrival(query:str, agent:str)** - Use this tool to retrieve and synthesize knowledge from a high-quality vector store of medical books and guidelines.  
   - Always pass the correct `agent` domain (Pathology).  
   - You may use it any number of times whenever deeper, authoritative medical knowledge is needed.  
   - If the first query does not provide a satisfactory answer, you may try **one or two re-phrased queries**, but do not enter an infinite loop.

    Guidelines:
    1. You are a helper Pathologist. **Primary Directive**: Your main job is to analyze lab reports. If a specialist gives you a command to "analyze" a sample (like a biopsy), your **first action MUST be to use the `ask_user` tool** to request the final lab report. You must treat the user as a **lab technician** who is providing you with the freshly generated results from the lab machines. 
    2. Take your time. Use all tools as needed. Tool outputs will return back to you.
    3. After framing a satisfactory answer for the specialist:
    - Call **add_report** to update your findings in the database.
    - In the **next turn**, return plain text only in this exact format:  
      `"This is the final report to specialist from Pathology labs: (Your final summary)"`
    4. If you do not use a tool when needed, or your plain output does not contain the exact phrase above, it will be ignored, and you will be asked to respond again. Avoid this mistake to prevent infinite loops.

    """)

    response = pathllm.invoke([SystemPrompt] + state['patho_messages'])
    return {'patho_messages': [response], 'current_agent': 'Pathologist'}


def router_patho(state: AgentState) -> AgentState:
    # Route the request to the appropriate pathologist agent
    last_message = state['patho_messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        ask_user_called = any(tc.get('name') == 'ask_user' for tc in last_message.tool_calls)
        for tool_call in last_message.tool_calls:
            if tool_call['name'] == 'add_report':
                report_content = tool_call['args']['report']
                state['current_report'].append(report_content)
        if ask_user_called:
            return "Patho_AskUser"
        return "Patho_Tooler"
    elif "final report" in last_message.content.lower() and "specialist" in last_message.content.lower():
        state["patho_QnA"].append("Pathologist Answer report to specialist:")
        state["patho_QnA"].append(last_message.content)
        # Safely route back to the caller; infer from history if stack is empty
        if state.get('next_agent') and len(state['next_agent']) > 0:
            return state['next_agent'].pop()
        # Infer caller from QnA trail
        caller_map = {
            'ophthalmologist': 'Ophthalmologist',
            'orthopedist': 'Orthopedist',
            'dermatologist': 'Dermatologist',
            'ent': 'ENT',
            'gynecologist': 'Gynecologist',
            'psychiatrist': 'Psychiatrist',
            'pediatrician': 'Pediatrician',
            'internal medicine': 'Internal Medicine',
        }
        for entry in reversed(state.get('patho_QnA') or []):
            low = entry.lower()
            for key, node in caller_map.items():
                if f"from {key}" in low:
                    return node
        # Last-resort: go back to a common specialist
        return "Orthopedist"
    else:
        return "Pathologist"

def Radiologist(state: AgentState) -> AgentState:
    # Implement the logic for the radiologist agent
    global patient_info
    callers = state.get('next_agent') or []
    caller = callers[-1] if callers else "General Physician"
    SystemPrompt = SystemMessage(content=f"""You are a High Quality Radiologist. You were called in by specialist {caller}.

    Patient data: {patient_info}

    Current context:
    - Total conversation with specialist: {", ".join(state['radio_QnA']) if state['radio_QnA'] else "None"}
    - If the last message is a question from the specialist, analyze it and frame your response. You may use tools if needed.

    Tools available:
    1. **ask_user** - Use this tool to ask the patient for imaging reports (X-rays, CT scans, etc.) or clarifying details.
    2. **search_internet** - Use this tool to search for any medical query related to interpreting images.
    3. **add_report** - Use this tool to add your final analysis of the imaging studies to the patient's record. Always call it before returning your final summary.
    
    Guidelines:
    1. **Your Role**: You are a helper Radiologist. Your primary task is to analyze imaging studies (like X-rays, CT scans, MRIs) based on the specialist's request. Use the `ask_user` tool to request these reports from the patient.
    2. **Pacing**: Take your time. Use all tools as needed. Tool outputs will always return to you for further analysis.
    3. **Reporting**: After you have analyzed the images and are ready to provide a satisfactory answer for the specialist:
        - First, call the **add_report** tool to log your detailed findings.
        - Then, in the **next turn**, return your conclusion as plain text in this exact format:
          `"This is the final report to specialist from Radiology labs: (Your final summary of the findings)"`
    4. **Error Handling**: If you do not use a tool or your plain text output doesn't match the required final report format, you will be prompted again. Avoid this to prevent loops.

    """)
    response = radllm.invoke([SystemPrompt] + state['radio_messages'])
    return {'radio_messages': [response], 'current_agent': 'Radiologist'}

def router_radio(state: AgentState) -> AgentState:
    # Route the request to the appropriate radiologist agent
    last_message = state['radio_messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        ask_user_called = any(tc.get('name') == 'ask_user' for tc in last_message.tool_calls)
        for tool_call in last_message.tool_calls:
            if tool_call['name'] == 'add_report':
                report_content = tool_call['args']['report']
                state['current_report'].append(report_content)
        if ask_user_called:
            return "Radio_AskUser"
        return "Radio_Tooler"
    elif "final report" in last_message.content.lower() and "specialist" in last_message.content.lower():
        state["radio_QnA"].append("Radiologist Answer report to specialist:")
        state["radio_QnA"].append(last_message.content)
        # Safely route back to the caller; infer from history if stack is empty
        if state.get('next_agent') and len(state['next_agent']) > 0:
            return state['next_agent'].pop()
        caller_map = {
            'ophthalmologist': 'Ophthalmologist',
            'orthopedist': 'Orthopedist',
            'dermatologist': 'Dermatologist',
            'ent': 'ENT',
            'gynecologist': 'Gynecologist',
            'psychiatrist': 'Psychiatrist',
            'pediatrician': 'Pediatrician',
            'internal medicine': 'Internal Medicine',
        }
        for entry in reversed(state.get('radio_QnA') or []):
            low = entry.lower()
            for key, node in caller_map.items():
                if f"from {key}" in low:
                    return node
        return "Orthopedist"
    else:
        return "Radiologist"

gp_tools = [Patient_data_report]
opthal_tools = [search_internet, add_report, VectorRAG_Retrival]
pedia_tools = [search_internet, add_report, VectorRAG_Retrival]
ortho_tools = [search_internet, add_report, VectorRAG_Retrival]
derma_tools = [search_internet, add_report, VectorRAG_Retrival]
ent_tools = [search_internet, add_report, VectorRAG_Retrival]
gynec_tools = [search_internet, add_report, VectorRAG_Retrival]
psych_tools = [search_internet, add_report, VectorRAG_Retrival]
med_tools = [search_internet, add_report, VectorRAG_Retrival]
patho_tools = [search_internet, add_report, VectorRAG_Retrival]
radio_tools = [search_internet, add_report]


opthal_tool_node = ToolNode(opthal_tools)

# The corrected "adapter" node for the graph
def opthal_tool_invoker(state: AgentState) -> dict:
    """
    Takes tool calls from 'specialist_messages', runs them,
    and returns the output to be added back to 'specialist_messages'.
    """
    # Create the dictionary input the ToolNode expects
    tool_input = {'messages': [state['specialist_messages'][-1]]}
    
    # Run the standard ToolNode. It returns a dictionary like {'messages': [ToolMessage(...)]}
    tool_output_dict = opthal_tool_node.invoke(tool_input)
    
    # *** THIS IS THE NEW LINE ***
    # We must extract the list of messages from that dictionary
    tool_output_messages = tool_output_dict['messages']
    
    # Return the list under the correct key for our specialist agent
    return {'specialist_messages': tool_output_messages}

agents = ["General Physician", "Pediatrics", "Ophthalmology", "Orthopedics", "Gastroenterology",
          "Radiology", "Pathology", "Dermatology", "ENT", "Gynecology", "Psychiatry", "Internal Medicine"]

opthal_tool_node = ToolNode(opthal_tools)
pedia_tool_node = ToolNode(pedia_tools)
ortho_tool_node = ToolNode(ortho_tools)
derma_tool_node = ToolNode(derma_tools)
ent_tool_node = ToolNode(ent_tools)
gynec_tool_node = ToolNode(gynec_tools)
psych_tool_node = ToolNode(psych_tools)
med_tool_node = ToolNode(med_tools)
patho_tool_node = ToolNode(patho_tools)
radio_tool_node = ToolNode(radio_tools)
gp_tool_node = ToolNode(gp_tools)

# Ask-user-only tool nodes (one per agent family), used to intercept and pause before execution
gp_ask_toolnode = ToolNode([ask_user])
opthal_ask_toolnode = ToolNode([ask_user])
pedia_ask_toolnode = ToolNode([ask_user])
ortho_ask_toolnode = ToolNode([ask_user])
derma_ask_toolnode = ToolNode([ask_user])
ent_ask_toolnode = ToolNode([ask_user])
gynec_ask_toolnode = ToolNode([ask_user])
psych_ask_toolnode = ToolNode([ask_user])
med_ask_toolnode = ToolNode([ask_user])
patho_ask_toolnode = ToolNode([ask_user])
radio_ask_toolnode = ToolNode([ask_user])


def opthal_tool_invoker(state: AgentState) -> dict:
    """
    Takes tool calls from 'specialist_messages', runs them,
    and returns the output to be added back to 'specialist_messages'.
    """
    tool_input = {'messages': [state['specialist_messages'][-1]]}
    tool_output_dict = opthal_tool_node.invoke(tool_input)
    tool_output_messages = tool_output_dict['messages']
    return {
        'specialist_messages': tool_output_messages,
        'current_agent': state.get('current_agent', 'Ophthalmologist')
    }

def opthal_askuser_invoker(state: AgentState) -> dict:
    tool_input = {'messages': [state['specialist_messages'][-1]]}
    last = state['specialist_messages'][-1]
    if isinstance(last, AIMessage):
        tool_input = {'messages': [last]}
        tool_output_dict = opthal_ask_toolnode.invoke(tool_input)
        return {
            'specialist_messages': tool_output_dict['messages'],
            'current_agent': state.get('current_agent', 'Ophthalmologist')
        }
    return {}

def derma_tool_invoker(state: AgentState) -> dict:
    """
    Takes tool calls from 'specialist_messages', runs them,
    and returns the output to be added back to 'specialist_messages'.
    """
    tool_input = {'messages': [state['specialist_messages'][-1]]}
    tool_output_dict = derma_tool_node.invoke(tool_input)
    tool_output_messages = tool_output_dict['messages']
    return {
        'specialist_messages': tool_output_messages,
        'current_agent': state.get('current_agent', 'Dermatologist')
    }

def derma_askuser_invoker(state: AgentState) -> dict:
    tool_input = {'messages': [state['specialist_messages'][-1]]}
    last = state['specialist_messages'][-1]
    if isinstance(last, AIMessage):
        tool_input = {'messages': [last]}
        tool_output_dict = derma_ask_toolnode.invoke(tool_input)
        return {
            'specialist_messages': tool_output_dict['messages'],
            'current_agent': state.get('current_agent', 'Dermatologist')
        }
    return {}

def pedia_tool_invoker(state: AgentState) -> dict:
    """
    Takes tool calls from 'specialist_messages', runs them,
    and returns the output to be added back to 'specialist_messages'.
    """
    tool_input = {'messages': [state['specialist_messages'][-1]]}
    tool_output_dict = pedia_tool_node.invoke(tool_input)
    tool_output_messages = tool_output_dict['messages']
    return {
        'specialist_messages': tool_output_messages,
        'current_agent': state.get('current_agent', 'Pediatrician')
    }

def pedia_askuser_invoker(state: AgentState) -> dict:
    last = state['specialist_messages'][-1]
    if isinstance(last, AIMessage):
        tool_input = {'messages': [last]}
        tool_output_dict = pedia_ask_toolnode.invoke(tool_input)
        return {
            'specialist_messages': tool_output_dict['messages'],
            'current_agent': state.get('current_agent', 'Pediatrician')
        }
    return {}

def ortho_tool_invoker(state: AgentState) -> dict:
    """
    Takes tool calls from 'specialist_messages', runs them,
    and returns the output to be added back to 'specialist_messages'.
    """
    tool_input = {'messages': [state['specialist_messages'][-1]]}
    tool_output_dict = ortho_tool_node.invoke(tool_input)
    tool_output_messages = tool_output_dict['messages']
    return {
        'specialist_messages': tool_output_messages,
        'current_agent': state.get('current_agent', 'Orthopedist')
    }

def ortho_askuser_invoker(state: AgentState) -> dict:
    last = state['specialist_messages'][-1]
    if isinstance(last, AIMessage):
        tool_input = {'messages': [last]}
        tool_output_dict = ortho_ask_toolnode.invoke(tool_input)
        return {
            'specialist_messages': tool_output_dict['messages'],
            'current_agent': state.get('current_agent', 'Orthopedist')
        }
    return {}

def ent_tool_invoker(state: AgentState) -> dict:
    """
    Takes tool calls from 'specialist_messages', runs them,
    and returns the output to be added back to 'specialist_messages'.
    """
    tool_input = {'messages': [state['specialist_messages'][-1]]}
    tool_output_dict = ent_tool_node.invoke(tool_input)
    tool_output_messages = tool_output_dict['messages']
    return {
        'specialist_messages': tool_output_messages,
        'current_agent': state.get('current_agent', 'ENT')
    }

def ent_askuser_invoker(state: AgentState) -> dict:
    tool_input = {'messages': [state['specialist_messages'][-1]]}
    last = state['specialist_messages'][-1]
    if isinstance(last, AIMessage):
        tool_input = {'messages': [last]}
        tool_output_dict = ent_ask_toolnode.invoke(tool_input)
        return {
            'specialist_messages': tool_output_dict['messages'],
            'current_agent': state.get('current_agent', 'ENT')
        }
    return {}

def gynec_tool_invoker(state: AgentState) -> dict:
    """
    Takes tool calls from 'specialist_messages', runs them,
    and returns the output to be added back to 'specialist_messages'.
    """
    tool_input = {'messages': [state['specialist_messages'][-1]]}
    tool_output_dict = gynec_tool_node.invoke(tool_input)
    tool_output_messages = tool_output_dict['messages']
    return {
        'specialist_messages': tool_output_messages,
        'current_agent': state.get('current_agent', 'Gynecologist')
    }

def gynec_askuser_invoker(state: AgentState) -> dict:
    tool_input = {'messages': [state['specialist_messages'][-1]]}
    last = state['specialist_messages'][-1]
    if isinstance(last, AIMessage):
        tool_input = {'messages': [last]}
        tool_output_dict = gynec_ask_toolnode.invoke(tool_input)
        return {
            'specialist_messages': tool_output_dict['messages'],
            'current_agent': state.get('current_agent', 'Gynecologist')
        }
    return {}

def psych_tool_invoker(state: AgentState) -> dict:
    """
    Takes tool calls from 'specialist_messages', runs them,
    and returns the output to be added back to 'specialist_messages'.
    """
    tool_input = {'messages': [state['specialist_messages'][-1]]}
    tool_output_dict = psych_tool_node.invoke(tool_input)
    tool_output_messages = tool_output_dict['messages']
    return {
        'specialist_messages': tool_output_messages,
        'current_agent': state.get('current_agent', 'Psychiatrist')
    }

def psych_askuser_invoker(state: AgentState) -> dict:
    tool_input = {'messages': [state['specialist_messages'][-1]]}
    last = state['specialist_messages'][-1]
    if isinstance(last, AIMessage):
        tool_input = {'messages': [last]}
        tool_output_dict = psych_ask_toolnode.invoke(tool_input)
        return {
            'specialist_messages': tool_output_dict['messages'],
            'current_agent': state.get('current_agent', 'Psychiatrist')
        }
    return {}

def med_tool_invoker(state: AgentState) -> dict:
    """
    Takes tool calls from 'specialist_messages', runs them,
    and returns the output to be added back to 'specialist_messages'.
    """
    tool_input = {'messages': [state['specialist_messages'][-1]]}
    tool_output_dict = med_tool_node.invoke(tool_input)
    tool_output_messages = tool_output_dict['messages']
    return {
        'specialist_messages': tool_output_messages,
        'current_agent': state.get('current_agent', 'Internal Medicine')
    }

def med_askuser_invoker(state: AgentState) -> dict:
    tool_input = {'messages': [state['specialist_messages'][-1]]}
    last = state['specialist_messages'][-1]
    if isinstance(last, AIMessage):
        tool_input = {'messages': [last]}
        tool_output_dict = med_ask_toolnode.invoke(tool_input)
        return {
            'specialist_messages': tool_output_dict['messages'],
            'current_agent': state.get('current_agent', 'Internal Medicine')
        }
    return {}

def patho_tool_invoker(state: AgentState) -> dict:
    """
    Takes tool calls from 'patho_messages', runs them,
    and returns the output to be added back to 'patho_messages'.
    """
    tool_input = {'messages': [state['patho_messages'][-1]]}
    tool_output_dict = patho_tool_node.invoke(tool_input)
    tool_output_messages = tool_output_dict['messages']
    return {
        'patho_messages': tool_output_messages,
        'current_agent': state.get('current_agent', 'Pathologist')
    }

def patho_askuser_invoker(state: AgentState) -> dict:
    tool_input = {'messages': [state['patho_messages'][-1]]}
    last = state['patho_messages'][-1]
    if isinstance(last, AIMessage):
        tool_input = {'messages': [last]}
        tool_output_dict = patho_ask_toolnode.invoke(tool_input)
        return {
            'patho_messages': tool_output_dict['messages'],
            'current_agent': state.get('current_agent', 'Pathologist')
        }
    return {}

def radio_tool_invoker(state: AgentState) -> dict:
    """
    Takes tool calls from 'radio_messages', runs them,
    and returns the output to be added back to 'radio_messages'.
    """
    tool_input = {'messages': [state['radio_messages'][-1]]}
    tool_output_dict = radio_tool_node.invoke(tool_input)
    tool_output_messages = tool_output_dict['messages']
    return {
        'radio_messages': tool_output_messages,
        'current_agent': state.get('current_agent', 'Radiologist')
    }

def radio_askuser_invoker(state: AgentState) -> dict:
    tool_input = {'messages': [state['radio_messages'][-1]]}
    last = state['radio_messages'][-1]
    if isinstance(last, AIMessage):
        tool_input = {'messages': [last]}
        tool_output_dict = radio_ask_toolnode.invoke(tool_input)
        return {
            'radio_messages': tool_output_dict['messages'],
            'current_agent': state.get('current_agent', 'Radiologist')
        }
    return {}



graph = StateGraph(AgentState)
graph.add_node("GP", general_physician)
graph.add_node("Ophthalmologist", Ophthalmologist)
graph.add_node("Pediatrician", Pediatrician)
graph.add_node("Orthopedist", Orthopedist)
graph.add_node("Dermatologist", Dermatologist)
graph.add_node("ENT", ENT)
graph.add_node("Gynecologist", Gynecologist)
graph.add_node("Psychiatrist", Psychiatrist)
graph.add_node("Internal Medicine", Internal_Medicine)

graph.add_node("Pathologist", Pathologist)
graph.add_node("Radiologist", Radiologist)
graph.add_node("Patho_Tooler", patho_tool_invoker)
graph.add_node("Radio_Tooler", radio_tool_invoker)
graph.add_node("Patho_AskUser", patho_askuser_invoker)
graph.add_node("Radio_AskUser", radio_askuser_invoker)
graph.add_edge("Patho_Tooler", "Pathologist")
graph.add_edge("Radio_Tooler", "Radiologist")

def gp_askuser_invoker(state: AgentState) -> dict:
    last = state['messages'][-1]
    if isinstance(last, AIMessage):
        tool_input = {'messages': [last]}
        tool_output_dict = gp_ask_toolnode.invoke(tool_input)
        return {'messages': tool_output_dict['messages'], 'current_agent': state.get('current_agent', 'GP')}
    return {}

def gp_tool_invoker(state: AgentState) -> dict:
    tool_input = {'messages': [state['messages'][-1]]}
    tool_output_dict = gp_tool_node.invoke(tool_input)
    return {'messages': tool_output_dict['messages'], 'current_agent': state.get('current_agent', 'GP')}

graph.add_node("GP_Tooler", gp_tool_invoker)
graph.add_node("GP_AskUser", gp_askuser_invoker)

graph.add_node("Ophthal_Tooler", opthal_tool_invoker)
graph.add_node("Pedia_Tooler", pedia_tool_invoker)
graph.add_node("Ortho_Tooler", ortho_tool_invoker)
graph.add_node("Dermat_Tooler", derma_tool_invoker)
graph.add_node("ENT_Tooler", ent_tool_invoker)
graph.add_node("Gynec_Tooler", gynec_tool_invoker)
graph.add_node("Psych_Tooler", psych_tool_invoker)
graph.add_node("IntMed_Tooler", med_tool_invoker)
graph.add_node("Ophthal_AskUser", opthal_askuser_invoker)
graph.add_node("Pedia_AskUser", pedia_askuser_invoker)
graph.add_node("Ortho_AskUser", ortho_askuser_invoker)
graph.add_node("Dermat_AskUser", derma_askuser_invoker)
graph.add_node("ENT_AskUser", ent_askuser_invoker)
graph.add_node("Gynec_AskUser", gynec_askuser_invoker)
graph.add_node("Psych_AskUser", psych_askuser_invoker)
graph.add_node("IntMed_AskUser", med_askuser_invoker)

graph.add_edge(START, "GP")
graph.add_edge("GP_Tooler", "GP")
graph.add_edge("Ophthal_Tooler", "Ophthalmologist")
graph.add_edge("Pedia_Tooler", "Pediatrician")
graph.add_edge("Ortho_Tooler", "Orthopedist")
graph.add_edge("Dermat_Tooler", "Dermatologist")
graph.add_edge("ENT_Tooler", "ENT")
graph.add_edge("Gynec_Tooler", "Gynecologist")
graph.add_edge("Psych_Tooler", "Psychiatrist")
graph.add_edge("IntMed_Tooler", "Internal Medicine")


graph.add_conditional_edges(
    "GP",
    router_gp,
    {
        "GP_AskUser": "GP_AskUser",
        "GP_Tooler": "GP_Tooler",
        "pediatrics": "Pediatrician",
        "Orthopedics": "Orthopedist",
        "ophthalmology": "Ophthalmologist",
        "dermatology": "Dermatologist",
        "ent": "ENT",
        "gynecology": "Gynecologist",
        "psychiatry": "Psychiatrist",
        "internal medicine": "Internal Medicine",
        "GP":"GP"
    }
)
graph.add_conditional_edges(
    "Ophthalmologist",
    router_opthal,
    {
        "Ophthal_AskUser": "Ophthal_AskUser",
        "Ophthal_Tooler": "Ophthal_Tooler",
        "Pathologist": "Pathologist",
        "Radiologist": "Radiologist",
        "Ophthalmologist": "Ophthalmologist",
        "end": END
    }
)
graph.add_conditional_edges(
    "Dermatologist",
    router_dermat,
    {
        "Dermat_AskUser": "Dermat_AskUser",
        "Dermat_Tooler": "Dermat_Tooler",
        "Pathologist": "Pathologist",
        "Radiologist": "Radiologist",
        "Dermatologist": "Dermatologist",
        "end": END
    }
)
graph.add_conditional_edges(
    "Pediatrician",
    router_pedia,
    {
        "Pedia_AskUser": "Pedia_AskUser",
        "Pedia_Tooler": "Pedia_Tooler",
        "Pathologist": "Pathologist",
        "Radiologist": "Radiologist",
        "Pediatrician": "Pediatrician",
        "end": END
    }
)
graph.add_conditional_edges(
    "Orthopedist",
    router_ortho,
    {
        "Ortho_AskUser": "Ortho_AskUser",
        "Ortho_Tooler": "Ortho_Tooler",
        "Pathologist": "Pathologist",
        "Radiologist": "Radiologist",
        "Orthopedist": "Orthopedist",
        "end": END
    }
)
graph.add_conditional_edges(
    "ENT",
    router_ent,
    {
        "ENT_AskUser": "ENT_AskUser",
        "ENT_Tooler": "ENT_Tooler",
        "Pathologist": "Pathologist",
        "Radiologist": "Radiologist",
        "ENT": "ENT",
        "end": END
    }
)
graph.add_conditional_edges(
    "Gynecologist",
    router_gynec,
    {
        "Gynec_AskUser": "Gynec_AskUser",
        "Gynec_Tooler": "Gynec_Tooler",
        "Pathologist": "Pathologist",
        "Radiologist": "Radiologist",
        "Gynecologist": "Gynecologist",
        "end": END
    }
)
graph.add_conditional_edges(
    "Psychiatrist",
    router_psych,
    {
        "Psych_AskUser": "Psych_AskUser",
        "Psych_Tooler": "Psych_Tooler",
        "Pathologist": "Pathologist",
        "Radiologist": "Radiologist",
        "Psychiatrist": "Psychiatrist",
        "end": END
    }
)
graph.add_conditional_edges(
    "Internal Medicine",
    router_medicine,
    {
        "IntMed_AskUser": "IntMed_AskUser",
        "IntMed_Tooler": "IntMed_Tooler",
        "Pathologist": "Pathologist",
        "Radiologist": "Radiologist",
        "Internal Medicine": "Internal Medicine",
        "end": END
    }
)

graph.add_conditional_edges(
    "Pathologist",
    router_patho,
    {
        "Patho_AskUser": "Patho_AskUser",
        "Patho_Tooler": "Patho_Tooler",
        'Pediatrician': 'Pediatrician',
        'Ophthalmologist': 'Ophthalmologist',
        'Orthopedist': 'Orthopedist',
        'Dermatologist': 'Dermatologist',
        'ENT': 'ENT',
        'Gynecologist': 'Gynecologist',
        'Psychiatrist': 'Psychiatrist',
        'Internal Medicine': 'Internal Medicine',
        "Pathologist": "Pathologist"
    }
)
graph.add_conditional_edges(
    "Radiologist",
    router_radio,
    {
        "Radio_AskUser": "Radio_AskUser",
        "Radio_Tooler": "Radio_Tooler",
        'Pediatrician': 'Pediatrician',
        'Ophthalmologist': 'Ophthalmologist',
        'Orthopedist': 'Orthopedist',
        'Dermatologist': 'Dermatologist',
        'ENT': 'ENT',
        'Gynecologist': 'Gynecologist',
        'Psychiatrist': 'Psychiatrist',
        'Internal Medicine': 'Internal Medicine',
        "Radiologist": "Radiologist"
    }
)
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
# Interrupt before every *_AskUser node so the API can collect input from frontend
myapp = graph.compile(
    interrupt_before=[
        "GP_AskUser","Ophthal_AskUser","Pedia_AskUser","Ortho_AskUser",
        "Dermat_AskUser","ENT_AskUser","Gynec_AskUser","Psych_AskUser",
        "IntMed_AskUser","Patho_AskUser","Radio_AskUser"
    ],
    checkpointer=memory
)

__all__ = ["myapp", "AgentState"]