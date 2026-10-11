
import numpy as np

from src.documents.reader import ler_pasta_pdfs
from src.search.retriever import BuscadorSemantico

trechos = ler_pasta_pdfs("books")
buscador = BuscadorSemantico(trechos)

pergunta = "Como Brás Cubas morreu?"

vetor = buscador.modelo.encode(
    pergunta,
    normalize_embeddings=True
)

semanticas = buscador.vetores @ vetor
lexicais = buscador.pontuar_bm25(pergunta)

for indice, trecho in enumerate(trechos):
    if (
        trecho["arquivo"] == "memorias_postumas_bras_cubas.pdf"
        and trecho["pagina"] == 9
        and "Morri de uma pneumonia" in trecho["texto"]
    ):
        print("\nRESULTADO DO TESTE")
        print("Página:", trecho["pagina"])

        print(
            "Posição semântica:",
            int(np.sum(semanticas > semanticas[indice]) + 1)
        )

        print(
            "Posição BM25:",
            int(np.sum(lexicais > lexicais[indice]) + 1)
        )
