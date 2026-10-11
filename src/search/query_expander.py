
import json
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()


def gerar_consultas(pergunta):
    pergunta = pergunta.strip()

    if not pergunta:
        return []

    chave = os.getenv("GEMINI_API_KEY")

    if not chave:
        return [pergunta]

    instrucao = f"""
Você é um especialista em recuperação de informações.

Sua tarefa é reformular uma pergunta para melhorar
a busca de trechos em documentos.

Regras:
1. Gere exatamente 3 consultas alternativas.
2. Preserve o significado da pergunta original.
3. Não responda à pergunta.
4. Não invente fatos, locais, nomes ou acontecimentos.
5. Não pressuponha que uma hipótese seja verdadeira.
6. Utilize português brasileiro.
7. Crie consultas com palavras e estruturas diferentes.
8. Retorne apenas JSON válido.

Formato:

{{
    "consultas": [
        "Primeira consulta",
        "Segunda consulta",
        "Terceira consulta"
    ]
}}

Pergunta original:
{pergunta}
"""

    try:
        cliente = genai.Client(api_key=chave)

        resposta = cliente.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=instrucao,
            config={
                "response_mime_type": "application/json",
                "temperature": 0.2
            }
        )

        if not resposta.text:
            return [pergunta]

        dados = json.loads(resposta.text)

        if not isinstance(dados, dict):
            return [pergunta]

        alternativas = dados.get("consultas", [])

        if not isinstance(alternativas, list):
            return [pergunta]

        consultas = [pergunta]

        for alternativa in alternativas:
            if not isinstance(alternativa, str):
                continue

            alternativa = alternativa.strip()

            if (
                alternativa
                and alternativa.casefold()
                not in {
                    consulta.casefold()
                    for consulta in consultas
                }
            ):
                consultas.append(alternativa)

            if len(consultas) >= 4:
                break

        return consultas

    except Exception as erro:
        print(
            f"Não foi possível reformular a pergunta: {erro}"
        )

        return [pergunta]
