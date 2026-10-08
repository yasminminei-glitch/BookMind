from pathlib import Path
import pymupdf

from .splitter import dividir_texto


def ler_pdf(caminho):
    caminho = Path(caminho)

    if not caminho.is_file():
        print("Erro: arquivo PDF não encontrado.")
        return []

    todos_trechos = []

    with pymupdf.open(caminho) as documento:
        print(f"Total de páginas: {len(documento)}")

        for numero_pagina, pagina in enumerate(documento, start=1):
            texto = pagina.get_text()

            trechos = dividir_texto(texto, numero_pagina)

            todos_trechos.extend(trechos)

    print(f"Total de trechos criados: {len(todos_trechos)}")

    if todos_trechos:
        print("\nPrimeiro trecho:")
        print(f"Página: {todos_trechos[0]['pagina']}")
        print(todos_trechos[0]["texto"][:500])

    return todos_trechos