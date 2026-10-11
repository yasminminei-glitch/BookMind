
from pathlib import Path

from .documents.reader import ler_pasta_pdfs
from .search.retriever import BuscadorSemantico
from .search.reranker import Reclassificador
from .assistant.responder import gerar_resposta


def main():
    print("================================")
    print("       BOOKMIND ASSISTANT       ")
    print("================================")

    pasta_projeto = Path(__file__).resolve().parent.parent
    pasta_livros = pasta_projeto / "books"

    print("\nCarregando documentos...")

    trechos = ler_pasta_pdfs(pasta_livros)

    if not trechos:
        print("Nenhum trecho encontrado nos documentos.")
        return

    arquivos = sorted({
        trecho["arquivo"]
        for trecho in trechos
    })

    print("\nDocumentos disponíveis:")

    for arquivo in arquivos:
        print(f"- {arquivo}")

    pergunta = input(
        "\nFaça uma pergunta sobre os documentos: "
    ).strip()

    if not pergunta:
        print("Nenhuma pergunta informada.")
        return

    print("\nPreparando busca...")

    buscador = BuscadorSemantico(trechos)

    # Recuperação ampla.
    candidatos = buscador.buscar(
        pergunta,
        limite=50
    )

    print(
        f"\nCandidatos recuperados: {len(candidatos)}"
    )

    print("\nReclassificando candidatos...")

    reclassificador = Reclassificador()

    resultados = reclassificador.reclassificar(
        pergunta,
        candidatos,
        limite=3
    )

    print("\nTrechos selecionados:")

    for resultado in resultados:
        print(
            f"\nArquivo: {resultado['arquivo']}"
        )
        print(
            f"Página principal: {resultado['pagina']}"
        )
        print(
            f"Pontuação do reranker: "
            f"{resultado['pontuacao_reranker']:.4f}"
        )
        print(resultado["texto"])
        print("--------------------")

    resposta = gerar_resposta(
        pergunta,
        resultados
    )

    print("\n================================")
    print("          RESPOSTA")
    print("================================")
    print(resposta)


if __name__ == "__main__":
    main()
