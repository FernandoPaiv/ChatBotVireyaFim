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
    """Retorna ou cria um histórico de chat da sessão."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_today_iso():
    """Retorna a data local ISO."""
    return datetime.now(TZ).date().isoformat()


def create_llm(api_key: str, model: str = "gemini-2.5-flash", temperature: float = 0.95):
    """Cria e retorna uma instância do modelo generativo."""
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key
    )


def create_llm_flash(api_key: str):
    """Cria o modelo flash auxiliar (para chains leves)."""
    return create_llm(api_key, model="gemini-2.0-flash", temperature=0.3)


def build_judge_chain(llm_flash):
    prompt = ChatPromptTemplate.from_messages([
        system_prompt_judge,
        ("human", "{usuario}")
    ]).partial(today_local=get_today_iso())

    return prompt | llm_flash | StrOutputParser()


def build_router_chain(llm_flash):
    prompt = ChatPromptTemplate.from_messages([
        system_prompt_roteador,
        fewshots_roteador,
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]).partial(today_local=get_today_iso())

    chain = prompt | llm_flash | StrOutputParser()
    return RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )


def build_rag_chain(llm_flash):
    prompt = ChatPromptTemplate.from_messages([
        system_prompt_rag,
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]).partial(today_local=get_today_iso())

    chain = prompt | llm_flash | StrOutputParser()
    return RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )


def build_mgr_assist_chain(llm, tools=TOOLS):
    prompt = ChatPromptTemplate.from_messages([
        system_prompt_eta_gerente,
        fewshots_eta_gerente,
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        (MessagesPlaceholder("agent_scratchpad"))
    ]).partial(today_local=get_today_iso())

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    return RunnableWithMessageHistory(
        executor,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )


def build_curador_chain(llm, tools=REDIS_TOOLS):
    prompt = ChatPromptTemplate.from_messages([
        system_prompt_curador,
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        (MessagesPlaceholder("agent_scratchpad"))
    ]).partial(today_local=get_today_iso())

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    return RunnableWithMessageHistory(
        executor,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )


def initialize_system(api_key: str):
    """Inicializa todo o sistema com a chave API informada."""

    llm = create_llm(api_key)
    llm_flash = create_llm_flash(api_key)

    return {
        "judge_chain": build_judge_chain(llm_flash),
        "router_chain": build_router_chain(llm_flash),
        "rag_chain": build_rag_chain(llm_flash),
        "mgr_assist_chain": build_mgr_assist_chain(llm),
        "curador_chain": build_curador_chain(llm),
        "llm": llm,
        "llm_flash": llm_flash
    }