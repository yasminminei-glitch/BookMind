
import json
import os
import re

from dotenv import load_dotenv
from google import genai

load_dotenv()


def fontes_disponiveis(resultados):
    fontes = set()

    for resultado in resultados:
        arquivo = resultado.get("arquivo")

        if not arquivo:
            continue

        paginas = resultado.get(
            "paginas",
            [resultado["pagina"]]
        )

        for pagina in paginas:
            fontes.add((arquivo, pagina))

    return fontes


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
        return (
            "Não encontrei informações suficientes "
            "nos documentos para responder."
        )

    chave = os.getenv("GEMINI_API_KEY")

    if not chave:
        return "Erro: chave da API do Gemini não encontrada."

    trechos = "\n\n".join(
        resultado["texto"]
        for resultado in resultados
    )

    fontes_validas = fontes_disponiveis(resultados)

    instrucao = f"""
Você é o assistente BookMind.

Responda à pergunta usando exclusivamente os trechos fornecidos.

Regras:
1. Não invente informações.
2. Se não houver evidência suficiente, admita isso.
3. Identifique somente as fontes que sustentam diretamente a resposta.
4. Cada fonte deve conter o nome exato do arquivo e o número da página.
5. Não inclua fontes apenas porque foram consultadas.
6. Não invente nomes de arquivos ou números de páginas.
7. Não misture informações de documentos diferentes.
8. Responda em português brasileiro.
9. Retorne exclusivamente um objeto JSON válido.
10. Não use blocos Markdown.

Formato obrigatório:

{{
    "resposta": "Sua resposta aqui",
    "fontes": [
        {{
            "arquivo": "documento.pdf",
            "pagina": 1
        }}
    ]
}}

Se não houver informações suficientes, utilize:

{{
    "resposta": "Não encontrei informações suficientes nos trechos fornecidos.",
    "fontes": []
}}

Pergunta:
{pergunta}

Trechos dos documentos:
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
        fontes_resposta = dados.get("fontes")

        if not isinstance(texto_resposta, str):
            return "A IA não retornou uma resposta textual válida."

        if not isinstance(fontes_resposta, list):
            return "A IA não retornou uma lista de fontes válida."

        fontes_citadas = set()

        for fonte in fontes_resposta:
            if not isinstance(fonte, dict):
                return "A IA retornou referências em formato inválido."

            arquivo = fonte.get("arquivo")
            pagina = fonte.get("pagina")

            if (
                not isinstance(arquivo, str)
                or not arquivo.strip()
                or type(pagina) is not int
            ):
                return "A IA retornou referências em formato inválido."

            fontes_citadas.add((arquivo, pagina))

        if not fontes_citadas.issubset(fontes_validas):
            return (
                "A IA citou fontes que não estavam nos trechos "
                "consultados. A resposta não foi apresentada."
            )

        texto_resposta = texto_resposta.strip()

        if not texto_resposta:
            return "A IA retornou uma resposta vazia."

        if not fontes_citadas:
            return (
                f"{texto_resposta}\n\n"
                "Fontes: nenhuma referência identificada."
            )

        referencias = "\n".join(
            f"- {arquivo} — página {pagina}"
            for arquivo, pagina in sorted(fontes_citadas)
        )

        return (
            f"{texto_resposta}\n\n"
            f"Fontes:\n{referencias}"
        )

    except Exception as erro:
        return f"Não foi possível consultar a IA: {erro}"
