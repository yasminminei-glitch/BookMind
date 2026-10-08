from pathlib import Path

from .documents.reader import ler_pdf
from .search.retriever import buscar_trechos
from .assistant.responder import gerar_resposta


def main():
    print("================================")
    print("       BOOKMIND ASSISTANT       ")
    print("================================")

    print("\nCarregando Dom Casmurro...")

    pasta_projeto = Path(__file__).resolve().parent.parent
    caminho_livro = pasta_projeto / "books" / "dom_casmurro.pdf"

    trechos = ler_pdf(caminho_livro)

    if not trechos:
        print("Nenhum trecho encontrado no livro.")
        return

    pergunta = input("\nFaça uma pergunta sobre o livro: ")

    resultados = buscar_trechos(pergunta, trechos)

    resposta = gerar_resposta(pergunta, resultados)

    print("\n================================")
    print("          RESPOSTA")
    print("================================")

    print(resposta)


if __name__ == "__main__":
    main()