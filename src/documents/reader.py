
from pathlib import Path
import re

import pymupdf

from .splitter import dividir_texto



def limpar_texto(texto):
    texto = re.sub(
        r"MEMÓRIAS PÓSTUMAS DE BRÁS CUBAS\s*[•\-–]\s*\d+\s*[•\-–]",
        " ",
        texto,
        flags=re.IGNORECASE
    )

    linhas = texto.splitlines()

    linhas_limpas = [
        linha.strip()
        for linha in linhas
        if linha.strip()
    ]

    return " ".join(linhas_limpas)


def ler_pdf(caminho):
    caminho = Path(caminho)

    if not caminho.is_file():
        print(f"Erro: PDF não encontrado: {caminho.name}")
        return []

    todos_trechos = []

    try:
        with pymupdf.open(caminho) as documento:
            print(f"\nLendo: {caminho.name}")
            print(f"Total de páginas: {len(documento)}")

            for numero_pagina, pagina in enumerate(
                documento,
                start=1
            ):
                texto = limpar_texto(pagina.get_text())

                if not texto.strip():
                    continue

                trechos = dividir_texto(
                    texto,
                    numero_pagina
                )

                for trecho in trechos:
                    trecho["arquivo"] = caminho.name

                todos_trechos.extend(trechos)

        print(f"Trechos criados: {len(todos_trechos)}")

    except (
        pymupdf.FileDataError,
        pymupdf.EmptyFileError
    ) as erro:
        print(f"Erro ao ler {caminho.name}: {erro}")
        return []

    return todos_trechos


def ler_pasta_pdfs(caminho_pasta):
    caminho_pasta = Path(caminho_pasta)

    if not caminho_pasta.is_dir():
        print("Erro: pasta de PDFs não encontrada.")
        return []

    arquivos = sorted(
        arquivo
        for arquivo in caminho_pasta.iterdir()
        if arquivo.is_file()
        and arquivo.suffix.lower() == ".pdf"
    )

    if not arquivos:
        print("Nenhum PDF encontrado na pasta.")
        return []

    todos_trechos = []

    print(f"\nEncontrados {len(arquivos)} arquivos PDF.")

    for arquivo in arquivos:
        trechos = ler_pdf(arquivo)
        todos_trechos.extend(trechos)

    print("\nLeitura concluída.")
    print(f"Total de trechos: {len(todos_trechos)}")

    return todos_trechos
