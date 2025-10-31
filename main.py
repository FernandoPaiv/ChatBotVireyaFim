import os
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from vector_search import buscar_similares
from chains import initialize_system
from utils import get_session_id, get_memories
from datetime import datetime


load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")

if not API_TOKEN:
    raise ValueError("⚠️ ERRO: variável de ambiente API_TOKEN não encontrada!")


app = FastAPI(title="ETA ChatBot API")

origins = [
    "http://localhost:5173",
    "https://seu-frontend-render.onrender.com",
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
    api_key: str
    
class ChatResponse(BaseModel):
    resposta: str
    origem: str


def fluxo_rag(chains, user_message):
    documents = buscar_similares(user_message)
    resposta = chains["rag_chain"].invoke(
        {
            "input": f"Mensagem do usuário: {user_message}\nDocumentos mais recomendados: {documents}"
        },
        config={"configurable": {"session_id": "RAG_SESSION"}},
    )
    return resposta


def fluxo_assesor(chains, user_message):
    resposta = chains["router_chain"].invoke(
        {"input": user_message},
        config={"configurable": {"session_id": "ROUTER_SESSION"}},
    )
    if "ROUTE=" in resposta:
        route = str(resposta).split("ROUTE=")[1].split("\n")[0]
        if "," in route:
            if "rag" in route:
                return "m,r", resposta
            elif "gerente" in route:
                return "m,g", resposta
        else:
            if "rag" in route:
                return "r", resposta
            elif "gerente" in route:
                return "g", resposta
            else:
                return "m", resposta
    else:
        return resposta


def fluxo_juiz(chains, pergunta, resposta):
    avaliacao = chains["judge_chain"].invoke({
        "usuario": pergunta,
        "resposta": resposta
    })
    return avaliacao


def fluxo_curador(chains, pergunta):
    curadoria = chains["curador_chain"].invoke(
        {"input": pergunta},
        config={"configurable": {"session_id": "CURADOR_SESSION"}},
    )
    return curadoria


from fastapi import HTTPException, status

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_token)])
async def chat_endpoint(data: ChatInput, email: str):
    """Endpoint principal que executa o fluxo completo"""
    try:
        user_input = data.user_message
        api_key = data.api_key
        
        try:
            session_id = get_session_id(email)
        except Exception:
            raise HTTPException(
                status_code=404,
                detail="Sessão não encontrada para este usuário."
            )

        try:
            memorias = get_memories(session_id)
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="Erro ao recuperar memórias da sessão."
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Erro na leitura dos dados de entrada: {str(e)}"
        )

    try:
        chains = initialize_system(api_key)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="API Key inválida ou erro ao inicializar componentes."
        )

    user_input = f"Memorias:{memorias}\nMensagem:{user_input}"

    try:
        resultado = fluxo_assesor(chains, user_input)
        rota = resultado[0]
        resposta = "\n".join(str(resultado[1]).split("\n")[1:])

        # --- Fluxos ---
        if rota == "m,r":
            curadoria = fluxo_curador(chains, f"{resposta}\nSessionID:{session_id}")
            resposta_rag = fluxo_rag(chains, resposta)
            conteudo = f"{curadoria}\n{resposta_rag}"

            final = chains["router_chain"].invoke(
                {"input": f"RESPOSTA_FINAL={conteudo}\nORIGEM=curadoria_rag"},
                config={"configurable": {"session_id": "ROUTER_SESSION"}},
            )
            return ChatResponse(resposta=final, origem="CURADORIA_RAG")

        elif rota == "m,g":
            curadoria = fluxo_curador(chains, f"{resposta}\nSessionID:{session_id}")
            resposta_gerente = chains["mgr_assist_chain"].invoke(
                {"input": resposta},
                config={"configurable": {"session_id": "GERENTE_SESSION"}},
            )
            conteudo = f"{curadoria}\n{resposta_gerente}"

            final = chains["router_chain"].invoke(
                {"input": f"RESPOSTA_FINAL={conteudo}\nORIGEM=curadoria_gerente"},
                config={"configurable": {"session_id": "ROUTER_SESSION"}},
            )
            return ChatResponse(resposta=final, origem="CURADORIA_GERENTE")

        elif rota == "r":
            resposta_rag = fluxo_rag(chains, resposta)
            juiz = fluxo_juiz(chains, resposta, resposta_rag)
            conteudo = f"{resposta_rag}\nAvaliação: {juiz}"

            final = chains["router_chain"].invoke(
                {"input": f"RESPOSTA_FINAL={conteudo}\nORIGEM=rag"},
                config={"configurable": {"session_id": "ROUTER_SESSION"}},
            )
            return ChatResponse(resposta=final, origem="RAG")

        elif rota == "g":
            resposta_gerente = chains["mgr_assist_chain"].invoke(
                {"input": resposta},
                config={"configurable": {"session_id": "GERENTE_SESSION"}},
            )
            final = chains["router_chain"].invoke(
                {"input": f"RESPOSTA_FINAL={resposta_gerente}\nORIGEM=gerente"},
                config={"configurable": {"session_id": "ROUTER_SESSION"}},
            )
            return ChatResponse(resposta=final, origem="GERENTE")

        elif rota == "m":
            final = fluxo_curador(chains, f"{resposta}\nSessionID:{session_id}")
            return ChatResponse(resposta=final, origem="CURADORIA")

        else:
            return ChatResponse(resposta=resultado, origem="ASSISTENTE")

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno ao processar fluxo: {str(e)}"
        )



@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.now()
    }

if __name__ == "__main__":
    print("🚀 API do ChatBot ETA iniciando em http://127.0.0.1:8000/docs ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
