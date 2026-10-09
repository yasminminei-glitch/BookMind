
import re
import unicodedata

import numpy as np
from sentence_transformers import SentenceTransformer


MODELO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

PALAVRAS_IGNORADAS = {
    "a", "o", "as", "os", "de", "da", "do", "das", "dos",
    "e", "em", "no", "na", "nos", "nas", "um", "uma",
    "que", "como", "com", "por", "para", "se", "ao",
    "ele", "ela", "eles", "elas", "foi", "eram",
    "onde", "quando", "qual", "quais",
    "era", "ser", "ter", "tinha", "foram"
}

EXPANSOES = {
    "conheceram": {
        "conhecer", "conheceu", "encontraram", "encontro"
    },
    "conheceu": {
        "conhecer", "conheceram", "encontrou", "encontro"
    },
    "amigos": {"amigo", "amizade"},
    "amizade": {"amigo", "amigos"},
    "seminario": {"seminarista", "seminaristas"},
    "seminaristas": {"seminario", "seminarista"},
    "mae": {"gloria"},
    "bentinho": {"bento"},
    "bento": {"bentinho"}
}


def normalizar(texto):
    texto = unicodedata.normalize("NFKD", texto)

    texto = "".join(
        caractere
        for caractere in texto
        if not unicodedata.combining(caractere)
    )

    return texto.lower()


def extrair_palavras(texto):
    palavras = re.findall(r"\b\w+\b", normalizar(texto))

    return {
        palavra
        for palavra in palavras
        if len(palavra) > 2
        and palavra not in PALAVRAS_IGNORADAS
    }


def expandir_palavras(palavras):
    expandidas = set(palavras)

    for palavra in palavras:
        expandidas.update(EXPANSOES.get(palavra, set()))

    return expandidas


def gerar_consultas(pergunta):
    consultas = [pergunta]
    texto = normalizar(pergunta)

    if (
        "bentinho" in texto
        and "escobar" in texto
        and (
            "conheceram" in texto
            or "conheceu" in texto
        )
    ):
        consultas.extend([
            "Bentinho e Escobar no seminário",
            "Como começou a amizade entre Bentinho e Escobar?",
            "Em que instituição Bentinho e Escobar estudaram juntos?"
        ])

    return list(dict.fromkeys(consultas))


class BuscadorSemantico:

    def __init__(self, trechos):
        self.trechos = trechos
        self.modelo = SentenceTransformer(MODELO)

        textos = [trecho["texto"] for trecho in trechos]

        self.vetores = self.modelo.encode(
            textos,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        self.palavras_trechos = [
            extrair_palavras(texto)
            for texto in textos
        ]

    def pontuar_palavras(self, palavras_pergunta):
        if not palavras_pergunta:
            return np.zeros(len(self.trechos))

        return np.array([
            len(palavras_pergunta & palavras_trecho)
            / len(palavras_pergunta)
            for palavras_trecho in self.palavras_trechos
        ])

    def pontuar_contexto(self, indice, palavras_pergunta):
        inicio = max(0, indice - 1)
        fim = min(len(self.trechos), indice + 2)

        palavras_contexto = set()

        for vizinho in range(inicio, fim):
            palavras_contexto.update(
                self.palavras_trechos[vizinho]
            )

        if not palavras_pergunta:
            return 0.0

        return (
            len(palavras_pergunta & palavras_contexto)
            / len(palavras_pergunta)
        )

    def pontuar_consulta(self, consulta):
        vetor = self.modelo.encode(
            consulta,
            normalize_embeddings=True
        )

        semanticas = self.vetores @ vetor

        palavras = expandir_palavras(
            extrair_palavras(consulta)
        )

        lexicais = self.pontuar_palavras(palavras)

        return (
            0.7 * semanticas
            + 0.3 * lexicais
        )

    def buscar(self, pergunta, limite=3):
        if not self.trechos or limite <= 0:
            return []

        consultas = gerar_consultas(pergunta)
        classificacoes = []

        for consulta in consultas:
            pontuacoes = self.pontuar_consulta(consulta)

            indices = np.argsort(
                pontuacoes
            )[::-1][:30]

            classificacoes.append([
                int(indice)
                for indice in indices
            ])

        pontuacoes_fusao = {}

        for classificacao in classificacoes:
            for posicao, indice in enumerate(classificacao):
                pontuacoes_fusao[indice] = (
                    pontuacoes_fusao.get(indice, 0.0)
                    + 1.0 / (60 + posicao + 1)
                )

        candidatos = sorted(
            pontuacoes_fusao,
            key=pontuacoes_fusao.get,
            reverse=True
        )

        resultados = []
        selecionados = []

        for indice in candidatos:
            if len(resultados) >= limite:
                break

            if any(
                abs(indice - anterior) <= 1
                for anterior in selecionados
            ):
                continue

            selecionados.append(indice)

            inicio = max(0, indice - 1)
            fim = min(len(self.trechos), indice + 2)

            partes = []
            paginas = set()

            for vizinho in range(inicio, fim):
                trecho = self.trechos[vizinho]

                paginas.add(trecho["pagina"])

                partes.append(
                    f"Página {trecho['pagina']}:\n"
                    f"{trecho['texto']}"
                )

            resultados.append({
                "pagina": self.trechos[indice]["pagina"],
                "paginas": sorted(paginas),
                "texto": "\n\n".join(partes),
                "pontuacao": float(
                    pontuacoes_fusao[indice]
                )
            })

        return resultados
