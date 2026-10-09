
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

chave = os.getenv("GEMINI_API_KEY")

if not chave:
    print("Erro: chave da API não encontrada.")
else:
    cliente = genai.Client(api_key=chave)

    resposta = cliente.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents="Quem escreveu Dom Casmurro?"
    )

    print(resposta.text)
