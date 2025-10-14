import os
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from vector_search import buscar_similares
from chains import chain_rag, chain_router, mgr_assist_chain, judge_chain

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
    allow_origins=origins,  # pode usar ["*"] se quiser liberar tudo (somente para teste!)
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
        if 'rag' in resposta.lower():
            return 'r'
        else:
            return 'g'
    else:
        return resposta


def fluxo_juiz(pergunta, resposta):
    avaliacao = judge_chain.invoke({
        "usuario": pergunta,
        "resposta": resposta
    })
    return avaliacao


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_token)])
async def chat_endpoint(data: ChatInput):
    user_input = data.user_message

    try:
        rota = fluxo_assesor(user_input)

        if rota == 'r':
            resposta = fluxo_rag(user_input)
            conteudo = resposta['output'] if isinstance(resposta, dict) else resposta
            juizpergunta = fluxo_juiz(user_input, conteudo)

            if 'mensagem correta' in juizpergunta.lower():
                conteudo = juizpergunta.lower().split('mensagem correta: ')[1]
                
            resposta_final = chain_router.invoke(
                {"input": f"RESPOSTA_FINAL={conteudo}\nORIGEM=rag"},
                config={'configurable': {"session_id": "ROUTER_SESSION"}},
            )
            final_text = resposta_final['output'] if isinstance(resposta_final, dict) else resposta_final
            return ChatResponse(resposta=final_text, origem="RAG")

        elif rota == 'g':
            resposta = mgr_assist_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "GERENTE_SESSION"}},
            )
            conteudo = resposta['output'] if isinstance(resposta, dict) else resposta

            resposta_final = chain_router.invoke(
                {"input": f"RESPOSTA_FINAL={conteudo}\nORIGEM=gerente"},
                config={'configurable': {"session_id": "ROUTER_SESSION"}},
            )
            final_text = resposta_final['output'] if isinstance(resposta_final, dict) else resposta_final
            return ChatResponse(resposta=final_text, origem="GERENTE")

        else:
            return ChatResponse(resposta=rota, origem="ASSISTENTE")

    except Exception as e:
        return ChatResponse(resposta=f"Erro ao processar: {e}", origem="ERRO")


if __name__ == "__main__":
    print("🚀 API do ChatBot ETA iniciando em http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)