from langchain_community.chat_message_histories import ChatMessageHistory
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import create_tool_calling_agent,AgentExecutor
from datetime import datetime
from zoneinfo import ZoneInfo
from prompts import system_prompt_judge,system_prompt_rag,system_prompt_roteador,fewshots_roteador,system_prompt_eta_gerente,fewshots_eta_gerente,system_prompt_curador
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    FewShotChatMessagePromptTemplate
)
from pg_tools import TOOLS
from redis_tools import REDIS_TOOLS
load_dotenv()

TZ=ZoneInfo('America/Sao_Paulo')
load_dotenv()
api_key = os.getenv('api_key')

today=datetime.now(TZ).date()
store={}
def get_session_history(session_id) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.95,
    google_api_key=api_key
)

llm_flash=ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    temperature=0.3,
    google_api_key=api_key
)

prompt_judge = ChatPromptTemplate.from_messages([
    system_prompt_judge,                  
    ("human", "{usuario}")
])
prompt_judge=prompt_judge.partial(today_local=today.isoformat())
judge_chain= prompt_judge | llm_flash | StrOutputParser()
prompt_router = ChatPromptTemplate.from_messages([
    system_prompt_roteador,                          
    fewshots_roteador,                            
    MessagesPlaceholder("chat_history"), 
    ("human", "{input}")
])
prompt_roteador=prompt_router.partial(today_local=today.isoformat())
router_chain= prompt_roteador | llm_flash | StrOutputParser()

chain_router = RunnableWithMessageHistory(
    router_chain,
    get_session_history=get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history'
)
prompt_rag = ChatPromptTemplate.from_messages([
    system_prompt_rag,                          # system prompt
    MessagesPlaceholder("chat_history"),    # memória
    ("human", "{input}")
])
prompt_rag=prompt_rag.partial(today_local=today.isoformat())
rag_chain= prompt_rag | llm_flash | StrOutputParser()

chain_rag = RunnableWithMessageHistory(
    rag_chain,
    get_session_history=get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history'
)
prompt_mgr_assist = ChatPromptTemplate.from_messages([
    system_prompt_eta_gerente,                          # system prompt
    fewshots_eta_gerente,                               # Shots human/ai 
    MessagesPlaceholder("chat_history"),    # memória
    ("human", "{input}"),
    (MessagesPlaceholder('agent_scratchpad'))                                    # user prompt
])

prompt_mgr_assist=prompt_mgr_assist.partial(today_local=today.isoformat())

mgr_assist_agent= create_tool_calling_agent(llm,TOOLS,prompt_mgr_assist)
mgr_assist_agent_executor= AgentExecutor.from_agent_and_tools(agent=mgr_assist_agent,tools=TOOLS,verbose=True)

mgr_assist_chain = RunnableWithMessageHistory(
    mgr_assist_agent_executor,
    get_session_history=get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history'
)

prompt_curador = ChatPromptTemplate.from_messages([
    system_prompt_curador,                        
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    (MessagesPlaceholder('agent_scratchpad'))                                   
])

prompt_curador=prompt_curador.partial(today_local=today.isoformat())


curador_agent= create_tool_calling_agent(llm,REDIS_TOOLS,prompt_curador)
curador_agent_executor= AgentExecutor.from_agent_and_tools(agent=curador_agent,tools=REDIS_TOOLS,verbose=True)

curador_chain = RunnableWithMessageHistory(
    curador_agent_executor,
    get_session_history=get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history'
)