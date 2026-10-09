
import json
import os
import re

from dotenv import load_dotenv
from google import genai

load_dotenv()


def paginas_disponiveis(resultados):
    paginas = set()

    for resultado in resultados:
        paginas.update(
            resultado.get("paginas", [resultado["pagina"]])
        )

    return sorted(paginas)


def extrair_json(texto):
    texto = texto.strip()

    texto = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        texto,
        flags=re.IGNORECASE
    )

    return json.loads(texto)


def gerar_resposta(pergunta, resultados):
    if not resultados:
        return "Não encontrei informações suficientes no livro para responder."

    chave = os.getenv("GEMINI_API_KEY")

    if not chave:
        return "Erro: chave da API do Gemini não encontrada."

    trechos = "\n\n".join(
        resultado["texto"]
        for resultado in resultados
    )

    paginas_validas = set(paginas_disponiveis(resultados))

    instrucao = f"""
Você é o assistente BookMind.

Responda à pergunta usando exclusivamente os trechos fornecidos.

Regras:
1. Não invente informações.
2. Se não houver evidência suficiente, admita isso.
3. Identifique somente as páginas que sustentam diretamente a resposta.
4. Não inclua páginas apenas porque foram consultadas.
5. Não invente números de páginas.
6. Responda em português brasileiro.
7. Retorne exclusivamente um objeto JSON válido.
8. Não use blocos Markdown.

Formato obrigatório:

{{
    "resposta": "Sua resposta aqui",
    "paginas": [1, 2]
}}

Se não houver informações suficientes, utilize:

{{
    "resposta": "Não encontrei informações suficientes nos trechos fornecidos.",
    "paginas": []
}}

Pergunta:
{pergunta}

Trechos do livro:
{trechos}
"""

    try:
        cliente = genai.Client(api_key=chave)

        resposta = cliente.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=instrucao,
            config={
                "response_mime_type": "application/json"
            }
        )

        if not resposta.text:
            return (
                "A IA não conseguiu gerar uma resposta. "
                "Nenhuma referência foi confirmada."
            )

        try:
            dados = extrair_json(resposta.text)
        except (json.JSONDecodeError, TypeError):
            return (
                "A IA retornou uma resposta em formato inválido. "
                "Nenhuma referência foi confirmada."
            )

        if not isinstance(dados, dict):
            return "A IA retornou um formato de resposta inesperado."

        texto_resposta = dados.get("resposta")
        paginas_resposta = dados.get("paginas")

        if not isinstance(texto_resposta, str):
            return "A IA não retornou uma resposta textual válida."

        if not isinstance(paginas_resposta, list):
            return "A IA não retornou uma lista de páginas válida."

        paginas_citadas = set()

        for pagina in paginas_resposta:
            if type(pagina) is not int:
                return "A IA retornou referências em formato inválido."

            paginas_citadas.add(pagina)

        if not paginas_citadas.issubset(paginas_validas):
            return (
                "A IA citou páginas que não estavam nos trechos "
                "consultados. A resposta não foi apresentada."
            )

        texto_resposta = texto_resposta.strip()

        if not texto_resposta:
            return "A IA retornou uma resposta vazia."

        if not paginas_citadas:
            return (
                f"{texto_resposta}\n\n"
                "Páginas utilizadas: nenhuma identificada."
            )

        referencias = ", ".join(
            str(pagina)
            for pagina in sorted(paginas_citadas)
        )

        return (
            f"{texto_resposta}\n\n"
            f"Páginas utilizadas: {referencias}"
        )

    except Exception as erro:
        return f"Não foi possível consultar a IA: {erro}"
