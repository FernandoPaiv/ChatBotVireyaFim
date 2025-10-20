from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Optional
import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()
import datetime

host = os.getenv('host')
porta = os.getenv('porta')
database = os.getenv('database')
senha = os.getenv('senha')
user = os.getenv('user')
def conectar():
    conn=psycopg2.connect(
        dbname=database,
        user=user,
        password=senha,
        host=host,
        port=porta
        )
    return conn



def get_prioridade(prioridade:str) -> int:
    try:
        conn=conectar()
        cursor=conn.cursor()
        query='SELECT id_prioridade FROM prioridade WHERE nivel=%s'
        cursor.execute(query,(prioridade,))
        id_prioridade=cursor.fetchone()[0]
    except Exception as e:
        conn.rollback()
        print(f"Exceção: {e}")
    return id_prioridade

def get_status(status:str) -> int:
    try:
        conn=conectar()
        cursor=conn.cursor()
        query='SELECT id_status FROM status WHERE status=%s'
        cursor.execute(query,(status,))
        id_status=cursor.fetchone()[0]
    except Exception as e:
        conn.rollback()
        print(f"Exceção: {e}")
    return id_status

def get_funcionario(email:str) -> int:
    try:
        conn=conectar()
        cursor=conn.cursor()
        query='SELECT id_funcionario FROM funcionario WHERE email=%s'
        cursor.execute(query,(email,))
        id_funcionario=cursor.fetchone()[0]
    except Exception as e:
        conn.rollback()
        print(f"Exceção: {e}")
    return id_funcionario


#tool verificar avisos:

# Nenhum argumento é necessário, mas ainda criamos um schema (boa prática)
class VerificarAvisosArgs(BaseModel):
    incluir_resolvidos: bool = Field(default=False, description="Se verdadeiro, inclui avisos resolvidos (id_status != 1 ou 2).")

@tool("verificar_avisos", args_schema=VerificarAvisosArgs)
def verificar_avisos(incluir_resolvidos: bool = False) -> list[str]:
    """
    Retorna os avisos ativos do sistema (id_status = 1 ou 2).
    Cada aviso vem com sua descrição, data de ocorrência e prioridade.
    """
    try:
        conn = conectar()
        cursor = conn.cursor()

        if incluir_resolvidos:
            cursor.execute("""
                SELECT descricao, data_ocorrencia, nivel
                FROM avisos
                JOIN Prioridade ON Prioridade.id_prioridade = Avisos.id_prioridade
            """)
        else:
            cursor.execute("""
                SELECT descricao, data_ocorrencia, nivel
                FROM avisos
                JOIN Prioridade ON Prioridade.id_prioridade = Avisos.id_prioridade
                WHERE id_status IN (1, 2)
            """)

        dados = cursor.fetchall()
        status = [
            f"Descrição: {d[0]} | Data: {d[1]} | Prioridade: {d[2]}"
            for d in dados
        ]
        return status or ["Nenhum aviso ativo encontrado."]

    except Exception as e:
        if conn:
            conn.rollback()
        return [f"Erro ao verificar avisos: {e}"]

    finally:
        try:
            cursor.close()
            conn.close()
        except:
            pass


#tool criar tarefa:
class CriarTarefaArgs(BaseModel):
    descricao: str = Field(..., description="Descrição clara da tarefa a ser criada.")
    prioridade: str = Field(..., description="Nível de prioridade da tarefa (ex: Alta, Média, Baixa).")
    funcionario: str = Field(..., description="Nome ou identificador do funcionário responsável.")
    status: Optional[str] = Field(default="pendente", description="Status inicial da tarefa (ex: pendente,  andamento, concluida).")

@tool("criar_tarefa", args_schema=CriarTarefaArgs)
def criar_tarefa(
    descricao: str,
    prioridade: str,
    funcionario: str,
    status: str = "pendente",
) -> dict:
    """
    Cria uma nova tarefa no sistema, vinculada a um funcionário e com prioridade definida.
    Pode ser chamada tanto manualmente quanto de forma autônoma após a detecção de avisos.
    """
    try:
        conn = conectar()
        cursor = conn.cursor()

        id_prioridade = get_prioridade(prioridade)
        id_status = get_status(status)
        id_funcionario = get_funcionario(funcionario)

        query = """
            INSERT INTO tarefa (descricao, data_criacao, id_prioridade, id_funcionario, id_status)
            VALUES (%s, NOW(), %s, %s, %s)
        """
        cursor.execute(query, (descricao, id_prioridade, id_funcionario, id_status))
        conn.commit()

        return {
            "status": "success",
            "message": f"Tarefa criada com sucesso para {funcionario}.",
            "dados": {
                "descricao": descricao,
                "prioridade": prioridade,
                "funcionario": funcionario,
                "status": status
            }
        }

    except Exception as e:
        conn.rollback()
        return {"status": "error", "message": f"Erro ao criar tarefa: {e}"}

    finally:
        try:
            cursor.close()
            conn.close()
        except:
            pass



class AdicionarAvisosArgs(BaseModel):
    descricao: str = Field(..., description="Descrição detalhada do aviso (ex: Falha no sistema de pagamentos).")
    id_eta: int = Field(..., description="Identificador da ETA (equipamento, estação ou setor relacionado ao aviso).")
    prioridade: str = Field(..., description="Prioridade do aviso (ex: Alta, Média, Baixa).")
    status: Optional[str] = Field(default="pendente", description="Status inicial do aviso (ex: pendente,  andamento, concluida).")
@tool("adicionar_avisos", args_schema=AdicionarAvisosArgs)
def adicionar_avisos(
    descricao: str,
    id_eta: int,
    prioridade: str,
    status: str = "pendente",
) -> dict:
    """
    Adiciona um novo aviso no sistema, vinculando-o a uma ETA específica e com prioridade definida.
    Pode ser usado quando o assistente identificar uma ocorrência relevante que precisa ser registrada.
    """
    try:
        conn = conectar()
        cursor = conn.cursor()

        id_prioridade = get_prioridade(prioridade)
        id_status = get_status(status)

        query = """
            INSERT INTO avisos (descricao, data_ocorrencia, id_eta, id_prioridade, id_status)
            VALUES (%s, NOW(), %s, %s, %s)
        """
        cursor.execute(query, (descricao, id_eta, id_prioridade, id_status))
        conn.commit()

        return {
            "status": "success",
            "message": f"Aviso adicionado com sucesso para ETA {id_eta}.",
            "dados": {
                "descricao": descricao,
                "prioridade": prioridade,
                "status": status,
                "id_eta": id_eta
            }
        }

    except Exception as e:
        conn.rollback()
        return {"status": "error", "message": f"Erro ao adicionar aviso: {e}"}

    finally:
        try:
            cursor.close()
            conn.close()
        except:
            pass


class ListarFuncionarioArgs(BaseModel):
    tarefas: Optional[bool] = Field(default=False, description="Se verdadeiro, inclui também as tarefas de cada funcionário.")

@tool("listar_funcionarios", args_schema=ListarFuncionarioArgs)
def listar_funcionarios(tarefas: bool = False) -> list[str]:
    """
    Lista todos os funcionários do sistema.
    Se 'tarefas' for True, inclui também as tarefas atribuídas a cada funcionário,
    com descrição, data de criação e prioridade.
    """
    conn = None
    cursor = None
    retorno = []
    try:
        conn = conectar()
        cursor = conn.cursor()

        if tarefas:
            query = '''
            SELECT nome, email, descricao, data_criacao, nivel
            FROM tarefa t
            JOIN funcionario f ON f.id_funcionario = t.id_funcionario
            JOIN prioridade p ON p.id_prioridade = t.id_prioridade
            ORDER BY f.nome, t.data_criacao DESC
            '''
        else:
            query = '''
            SELECT nome, email
            FROM funcionario
            ORDER BY nome
            '''
        cursor.execute(query)
        dados = cursor.fetchall()

        if not dados:
            return ["Nenhum funcionário encontrado."]

        for i in dados:
            retorno.append(f"Funcionário:\nNome: {i[0]} | Email: {i[1]}")
            if tarefas:
                retorno.append(f"Tarefa:\nDescrição: {i[2]} | Data início: {i[3]} | Prioridade: {i[4]}")

        return retorno

    except Exception as e:
        if conn:
            conn.rollback()
        return [f"Erro ao listar funcionários: {e}"]

    finally:
        try:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
        except:
            pass

class ListarTarefaArgs(BaseModel):
    desc: Optional[str] = Field(default=None, description="Palavra-chave na descrição da tarefa.")
    email: Optional[str] = Field(default=None, description="E-mail ou nome do funcionário responsável.")
    datacriacao: Optional[datetime.date] = Field(default=None, description="Data de criação da tarefa (formato YYYY-MM-DD).")
    dataconclusao: Optional[datetime.date] = Field(default=None, description="Data de conclusão da tarefa (formato YYYY-MM-DD).")
    nivel: Optional[str] = Field(default=None, description="Nível de prioridade da tarefa (por exemplo: ALTA, MÉDIA, BAIXA).")

@tool("listar_tarefas", args_schema=ListarTarefaArgs)
def listar_tarefas(
    desc: Optional[str] = None,
    email: Optional[str] = None,
    datacriacao: Optional[datetime.date] = None,
    dataconclusao: Optional[datetime.date] = None,
    nivel: Optional[str] = None
) -> list[str]:
    """
    Lista as tarefas cadastradas com base nos filtros opcionais fornecidos:
    - descrição
    - e-mail do funcionário
    - data de criação
    - data de conclusão
    - nível de prioridade

    Retorna uma lista formatada com as principais informações de cada tarefa.
    """
    conn = None
    cursor = None
    try:
        conn = conectar()
        cursor = conn.cursor()

        base_query = '''
        SELECT descricao, data_criacao, data_conclusao, nivel, email
        FROM tarefa t
        JOIN funcionario f ON f.id_funcionario = t.id_funcionario
        JOIN prioridade p ON p.id_prioridade = t.id_prioridade
        '''
        filtros = []
        params = []

        if desc:
            filtros.append("descricao ILIKE %s")
            params.append(f"%{desc}%")
        if email:
            filtros.append("email ILIKE %s")
            params.append(f"%{email}%")
        if datacriacao:
            filtros.append("CAST(data_criacao AS DATE) = %s")
            params.append(datacriacao)
        if dataconclusao:
            filtros.append("CAST(data_conclusao AS DATE) = %s")
            params.append(dataconclusao)
        if nivel:
            filtros.append("nivel ILIKE %s")
            params.append(f"%{nivel}%")

        if filtros:
            base_query += " WHERE " + " AND ".join(filtros)

        base_query += " ORDER BY data_criacao DESC"

        cursor.execute(base_query, tuple(params))
        dados = cursor.fetchall()

        if not dados:
            return ["Nenhuma tarefa encontrada com os filtros aplicados."]

        tarefas = [
            f"Descrição: {i[0]} | Criada em: {i[1]} | Concluída em: {i[2]} | Prioridade: {i[3]} | Funcionário: {i[4]}"
            for i in dados
        ]
        return tarefas

    except Exception as e:
        if conn:
            conn.rollback()
        return [f"Erro ao listar tarefas: {e}"]

    finally:
        try:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
        except:
            pass

class AtualizarTarefaArgs(BaseModel):
    desc: str = Field(..., description="Palavra-chave presente na descrição da tarefa, como 'verificar' ou 'qualidade da água'.")
    email_func: str = Field(..., description="e-mail do funcionário responsável pela tarefa.")

@tool("atualizar_tarefa", args_schema=AtualizarTarefaArgs)
def atualizar_tarefa(desc: str, email_func: str) -> str:
    """
    Marca uma tarefa como concluída no banco de dados, com base em uma palavra-chave na descrição e no e-mail (ou nome) do funcionário.

    O agente gerente deve chamar esta tool quando detectar que uma tarefa foi finalizada ou concluída,
    mesmo que a descrição mencionada seja apenas parcial.
    """
    conn = None
    cursor = None
    try:
        id_funcionario=get_funcionario(email_func)
        conn = conectar()
        cursor = conn.cursor()
        query = '''
        UPDATE tarefa
        SET data_conclusao = NOW()
        WHERE descricao LIKE %s
        AND id_funcionario = %s
        AND data_conclusao IS NULL
        '''
        params = [f'%{desc}%',id_funcionario]
        cursor.execute(query, params)
        conn.commit()

        if cursor.rowcount > 0:
            return f"Tarefa contendo '{desc}' atribuída a '{email_func}' foi marcada como concluída."
        else:
            return f"Nenhuma tarefa ativa encontrada com descrição semelhante a '{desc}' para '{email_func}'."

    except Exception as e:
        if conn:
            conn.rollback()
        return f"Erro ao atualizar tarefa: {e}"

    finally:
        try:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
        except:
            pass
TOOLS = [
    verificar_avisos,
    criar_tarefa,
    adicionar_avisos,
    listar_funcionarios,
    listar_tarefas,
    atualizar_tarefa
]