import os
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from vector_search import buscar_similares
from chains import chain_rag, chain_router, mgr_assist_chain, judge_chain,curador_chain
from utils import get_session_id,get_memories

load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")

if not API_TOKEN:
    raise ValueError("⚠️ ERRO: variável de ambiente API_TOKEN não encontrada!")

app = FastAPI(title="ETA ChatBot API")

origins = [
    "http://localhost:5173",  # frontend local
    "https://seu-frontend-render.onrender.com",  # frontend hospedado no Render, se tiver
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verifica se o token Bearer é válido"""
    token = credentials.credentials
    if token != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido ou ausente.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

class ChatInput(BaseModel):
    user_message: str

class ChatResponse(BaseModel):
    resposta: str
    origem: str

def fluxo_rag(user_message):
    documents=buscar_similares(user_message)
    resposta = chain_rag.invoke(
        {
            "input": f"Mensagem do usuário: {user_message}\nDocumentos mais recomendados: {documents}"
        },
        config={'configurable': {"session_id": "RAG_SESSION"}},
    )
    return resposta


def fluxo_assesor(user_message):
    resposta = chain_router.invoke(
        {"input": user_message},
        config={'configurable': {"session_id": "ROUTER_SESSION"}},
    )
    if 'ROUTE=' in resposta:
        route=str(resposta).split('ROUTE=')[1].split('\n')[0]
        if ',' in route:
            if 'rag' in route:
                return 'm,r',resposta
            elif 'gerente' in route:
                return 'm,g',resposta
        else:
            if 'rag' in route:
                return 'r',resposta
            elif 'gerente' in route:
                return 'g',resposta
            else:
                return 'm',resposta


def fluxo_juiz(pergunta, resposta):
    avaliacao = judge_chain.invoke({
        "usuario": pergunta,
        "resposta": resposta
    })
    return avaliacao

def fluxo_curardor(pergunta):
    curadoria = curador_chain.invoke(
        {"input": pergunta},
        config={'configurable': {"session_id": "ROUTER_SESSION"}},
    )
    return curadoria


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_token)])
async def chat_endpoint(data: ChatInput, email: str):
    user_input = data.user_message
    session_id = get_session_id(email)
    memorias = get_memories(session_id)

    # Monta o prompt inicial
    user_input = f'Memorias:{memorias}\n{user_input}'
    user_input = f'Mensagem:{user_input}'

    try:
        resultado = fluxo_assesor(user_input)
        rota = resultado[0]
        resposta = '\n'.join(str(resultado[1]).split('\n')[1:])  # limpa a primeira linha

        # --- ROTAS MÚLTIPLAS ---
        if rota == 'm,r':
            curadoria = fluxo_curardor(f'{resposta}\nSessionID:{session_id}')
            resposta_rag = fluxo_rag(resposta)
            conteudo = resposta_rag['output'] if isinstance(resposta_rag, dict) else resposta_rag
            conteudo_final = f"{curadoria}\n{conteudo}"

            resposta_final = chain_router.invoke(
                {"input": f"RESPOSTA_FINAL={conteudo_final}\nORIGEM=curadoria_rag"},
                config={'configurable': {"session_id": "ROUTER_SESSION"}},
            )
            final_text = resposta_final['output'] if isinstance(resposta_final, dict) else resposta_final
            return ChatResponse(resposta=final_text, origem="CURADORIA_RAG")

        elif rota == 'm,g':
            curadoria = fluxo_curardor(f'{resposta}\nSessionID:{session_id}')
            resposta_gerente = mgr_assist_chain.invoke(
                {"input": resposta},
                config={'configurable': {"session_id": "GERENTE_SESSION"}},
            )
            conteudo = resposta_gerente['output'] if isinstance(resposta_gerente, dict) else resposta_gerente
            conteudo_final = f"{curadoria}\n{conteudo}"

            resposta_final = chain_router.invoke(
                {"input": f"RESPOSTA_FINAL={conteudo_final}\nORIGEM=curadoria_gerente"},
                config={'configurable': {"session_id": "ROUTER_SESSION"}},
            )
            final_text = resposta_final['output'] if isinstance(resposta_final, dict) else resposta_final
            return ChatResponse(resposta=final_text, origem="CURADORIA_GERENTE")

        # --- ROTA RAG ---
        elif rota == 'r':
            resposta_rag = fluxo_rag(resposta)
            conteudo = resposta_rag['output'] if isinstance(resposta_rag, dict) else resposta_rag
            juiz = fluxo_juiz(resposta,conteudo)
            conteudo_final = f"{conteudo}\nAvaliação: {juiz}"

            resposta_final = chain_router.invoke(
                {"input": f"RESPOSTA_FINAL={conteudo_final}\nORIGEM=rag"},
                config={'configurable': {"session_id": "ROUTER_SESSION"}},
            )
            final_text = resposta_final['output'] if isinstance(resposta_final, dict) else resposta_final
            return ChatResponse(resposta=final_text, origem="RAG")

        # --- ROTA GERENTE ---
        elif rota == 'g':
            resposta_gerente = mgr_assist_chain.invoke(
                {"input": resposta},
                config={'configurable': {"session_id": "GERENTE_SESSION"}},
            )
            conteudo = resposta_gerente['output'] if isinstance(resposta_gerente, dict) else resposta_gerente

            resposta_final = chain_router.invoke(
                {"input": f"RESPOSTA_FINAL={conteudo}\nORIGEM=gerente"},
                config={'configurable': {"session_id": "ROUTER_SESSION"}},
            )
            final_text = resposta_final['output'] if isinstance(resposta_final, dict) else resposta_final
            return ChatResponse(resposta=final_text, origem="GERENTE")
        
        # --- SOMENTE CURADORIA ---
        elif rota == 'm':
            final_text = fluxo_curardor(f'{resposta}\nSessionID:{session_id}')
            return ChatResponse(resposta=final_text, origem="CURADORIA")

        else:
            return ChatResponse(resposta="Fluxo padrão (assistente)", origem="ASSISTENTE")

    except Exception as e:
        return ChatResponse(resposta=f"Erro ao processar fluxo: {e}", origem="ERRO")


if __name__ == "__main__":
    print("🚀 API do ChatBot ETA iniciando em http://127.0.0.1:8000/docs ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)