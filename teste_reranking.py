
from pathlib import Path

from sentence_transformers import CrossEncoder

from src.documents.reader import ler_pdf
from src.search.retriever import BuscadorSemantico


MODELO = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def encontrar_trecho_central(resultado, trechos):
    pagina = resultado["pagina"]
    texto_expandido = resultado["texto"]

    candidatos = []

    for trecho in trechos:
        if trecho["pagina"] != pagina:
            continue

        if trecho["texto"] in texto_expandido:
            candidatos.append(trecho)

    if not candidatos:
        return resultado["texto"]

    return max(
        candidatos,
        key=lambda trecho: len(trecho["texto"])
    )["texto"]


def main():
    pasta_projeto = Path(__file__).resolve().parent
    caminho_pdf = pasta_projeto / "books" / "dom_casmurro.pdf"

    print("Carregando livro...", flush=True)

    trechos = ler_pdf(caminho_pdf)

    if not trechos:
        print("Nenhum trecho encontrado.")
        return

    pergunta = "Quem foi Bentinho?"

    print("\nPreparando busca semântica...", flush=True)

    buscador = BuscadorSemantico(trechos)
    candidatos = buscador.buscar(pergunta, limite=10)

    if not candidatos:
        print("Nenhum candidato encontrado.")
        return

    print("\nPreparando trechos individuais...", flush=True)

    textos_centrais = [
        encontrar_trecho_central(resultado, trechos)
        for resultado in candidatos
    ]

    print("\nCarregando reclassificador...", flush=True)

    modelo = CrossEncoder(MODELO)

    pares = [
        (pergunta, texto)
        for texto in textos_centrais
    ]

    pontuacoes = modelo.predict(pares)

    classificados = sorted(
        zip(candidatos, textos_centrais, pontuacoes),
        key=lambda item: float(item[2]),
        reverse=True
    )

    print("\n" + "=" * 50)
    print("BUSCA ORIGINAL")
    print("=" * 50)

    for posicao, (resultado, texto) in enumerate(
        zip(candidatos, textos_centrais),
        start=1
    ):
        print(f"{posicao}. Página {resultado['pagina']}")
        print(texto[:350])
        print("--------------------")

    print("\n" + "=" * 50)
    print("RERANKING COM TRECHOS INDIVIDUAIS")
    print("=" * 50)

    for posicao, (resultado, texto, pontuacao) in enumerate(
        classificados,
        start=1
    ):
        print(
            f"{posicao}. Página {resultado['pagina']} "
            f"| Pontuação: {float(pontuacao):.4f}"
        )
        print(texto[:350])
        print("--------------------")


if __name__ == "__main__":
    main()
