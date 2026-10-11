
from sentence_transformers import CrossEncoder


MODELO_RERANKER = (
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)


class Reclassificador:
    def __init__(self):
        self.modelo = CrossEncoder(MODELO_RERANKER)

    def reclassificar(self, pergunta, candidatos, limite=3):
        if not candidatos:
            return []

        pares = [
            (pergunta, candidato["texto"])
            for candidato in candidatos
        ]

        pontuacoes = self.modelo.predict(pares)

        classificados = []

        for candidato, pontuacao in zip(
            candidatos,
            pontuacoes
        ):
            resultado = candidato.copy()
            resultado["pontuacao_reranker"] = float(
                pontuacao
            )
            classificados.append(resultado)

        classificados.sort(
            key=lambda item: item["pontuacao_reranker"],
            reverse=True
        )

        return classificados[:limite]
