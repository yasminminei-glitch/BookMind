
import math
import re
import unicodedata
from collections import Counter

import numpy as np
from nltk.stem import RSLPStemmer
from sentence_transformers import SentenceTransformer

from .query_expander import gerar_consultas


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
    "conheceram": {"conhecer", "conheceu", "encontraram", "encontro"},
    "conheceu": {"conhecer", "conheceram", "encontrou", "encontro"},
    "amigos": {"amigo", "amizade"},
    "amizade": {"amigo", "amigos"},
    "seminario": {"seminarista", "seminaristas"},
    "seminaristas": {"seminario", "seminarista"},
    "mae": {"gloria"},
    "bentinho": {"bento"},
    "bento": {"bentinho"}
}

STEMMER = RSLPStemmer()


def normalizar(texto):
    texto = unicodedata.normalize("NFKD", texto)

    return "".join(
        caractere
        for caractere in texto
        if not unicodedata.combining(caractere)
    ).lower()


def extrair_lista_palavras(texto):
    palavras = re.findall(
        r"\b\w+\b",
        normalizar(texto)
    )

    return [
        palavra
        for palavra in palavras
        if len(palavra) > 2
        and palavra not in PALAVRAS_IGNORADAS
    ]


def extrair_palavras(texto):
    return set(extrair_lista_palavras(texto))


def expandir_palavras(palavras):
    expandidas = set(palavras)

    for palavra in palavras:
        expandidas.update(
            EXPANSOES.get(palavra, set())
        )

    return expandidas


def extrair_raizes(texto):
    return {
        STEMMER.stem(palavra)
        for palavra in expandir_palavras(
            extrair_palavras(texto)
        )
    }


def extrair_lista_raizes(texto):
    return [
        STEMMER.stem(palavra)
        for palavra in extrair_lista_palavras(texto)
    ]


def parece_material_editorial(texto):
    texto_normalizado = normalizar(texto)

    indicadores = (
        "isbn",
        "catalogacao-na-publicacao",
        "catalogacao na publicacao",
        "direitos reservados",
        "editor responsavel",
        "diretoria-geral",
        "secretaria-geral",
        "ficha catalografica",
        "coordenacao edicoes",
        "projeto grafico",
        "preparacao de originais"
    )

    encontrados = sum(
        indicador in texto_normalizado
        for indicador in indicadores
    )

    return encontrados >= 2


class BuscadorSemantico:

    def __init__(self, trechos):
        self.trechos = trechos

        self.modelo = SentenceTransformer(MODELO)

        textos = [
            trecho["texto"]
            for trecho in trechos
        ]

        self.vetores = self.modelo.encode(
            textos,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        self.tokens = [
            extrair_lista_raizes(texto)
            for texto in textos
        ]

        self.frequencias = [
            Counter(tokens)
            for tokens in self.tokens
        ]

        self.tamanhos = np.array([
            len(tokens)
            for tokens in self.tokens
        ])

        self.tamanho_medio = (
            float(np.mean(self.tamanhos))
            if len(self.tamanhos)
            else 1.0
        )

        self.frequencia_documentos = Counter()

        for tokens in self.tokens:
            self.frequencia_documentos.update(
                set(tokens)
            )

        self.arquivos = [
            trecho.get("arquivo", "")
            for trecho in trechos
        ]

        self.editoriais = np.array([
            parece_material_editorial(
                trecho["texto"]
            )
            for trecho in trechos
        ])

    def pontuar_bm25(self, consulta):
        palavras = extrair_raizes(consulta)

        pontuacoes = np.zeros(
            len(self.trechos)
        )

        total = len(self.trechos)

        k1 = 1.5
        b = 0.75

        for palavra in palavras:
            frequencia_documentos = (
                self.frequencia_documentos.get(
                    palavra,
                    0
                )
            )

            if frequencia_documentos == 0:
                continue

            idf = math.log(
                1 + (
                    total - frequencia_documentos + 0.5
                ) / (
                    frequencia_documentos + 0.5
                )
            )

            for indice, frequencias in enumerate(
                self.frequencias
            ):
                frequencia = frequencias.get(
                    palavra,
                    0
                )

                if frequencia == 0:
                    continue

                denominador = (
                    frequencia
                    + k1 * (
                        1 - b
                        + b * self.tamanhos[indice]
                        / max(self.tamanho_medio, 1)
                    )
                )

                pontuacoes[indice] += (
                    idf
                    * frequencia
                    * (k1 + 1)
                    / denominador
                )

        return pontuacoes

    def identificar_arquivo(self, pergunta):
        palavras_pergunta = extrair_raizes(
            pergunta
        )

        candidatos = {}

        for arquivo in set(self.arquivos):
            nome = (
                arquivo.rsplit(".", 1)[0]
                .replace("_", " ")
                .replace("-", " ")
            )

            palavras_nome = extrair_raizes(
                nome
            )

            if not palavras_nome:
                continue

            correspondencias = len(
                palavras_nome & palavras_pergunta
            )

            if correspondencias >= 2:
                candidatos[arquivo] = (
                    correspondencias
                    / len(palavras_nome)
                )

        if not candidatos:
            return None

        ordenados = sorted(
            candidatos.items(),
            key=lambda item: item[1],
            reverse=True
        )

        if (
            len(ordenados) > 1
            and ordenados[0][1] == ordenados[1][1]
        ):
            return None

        return ordenados[0][0]

    def classificar_consulta(
        self,
        consulta,
        arquivo=None,
        termos_foco=None
    ):
        vetor = self.modelo.encode(
            consulta,
            normalize_embeddings=True
        )

        semanticas = self.vetores @ vetor

        lexicais = self.pontuar_bm25(
            consulta
        )

        if termos_foco:
            pontuacao_foco = self.pontuar_bm25(
                " ".join(termos_foco)
            )

            lexicais = (
                lexicais
                + 2.0 * pontuacao_foco
            )

        permitidos = np.logical_not(
            self.editoriais
        )

        if arquivo:
            permitidos &= np.array([
                nome == arquivo
                for nome in self.arquivos
            ])

        indices = np.flatnonzero(
            permitidos
        )

        if not len(indices):
            return (
                np.array([], dtype=int),
                np.array([], dtype=int)
            )

        classificacao_semantica = indices[
            np.argsort(
                semanticas[indices]
            )[::-1][:100]
        ]

        # Não incluir resultados com BM25 igual a zero.
        indices_lexicais = indices[
            lexicais[indices] > 0
        ]

        classificacao_lexical = (
            indices_lexicais[
                np.argsort(
                    lexicais[indices_lexicais]
                )[::-1][:100]
            ]
        )

        return (
            classificacao_semantica,
            classificacao_lexical
        )

    def diagnosticar_ranking(
        self,
        consultas,
        arquivo_alvo,
        termos_foco,
        candidatos,
        pontuacoes_fusao
    ):
        """
        Diagnóstico temporário para investigar
        a recuperação da página 9.
        """

        print(
            "\n=== DIAGNÓSTICO DO RANKING ==="
        )

        arquivo_teste = (
            "memorias_postumas_bras_cubas.pdf"
        )

        pagina_teste = 9

        indices_teste = [
            indice
            for indice, trecho in enumerate(
                self.trechos
            )
            if (
                trecho.get("arquivo") == arquivo_teste
                and trecho.get("pagina") == pagina_teste
                and "pneumonia" in normalizar(
                    trecho["texto"]
                )
            )
        ]

        if not indices_teste:
            print(
                "Nenhum trecho de teste encontrado."
            )
            print(
                "=== FIM DO DIAGNÓSTICO ===\n"
            )
            return

        rankings = []

        for consulta in consultas:
            semantica, lexical = (
                self.classificar_consulta(
                    consulta,
                    arquivo=arquivo_alvo,
                    termos_foco=termos_foco
                )
            )

            posicoes_semanticas = {
                int(indice): posicao
                for posicao, indice in enumerate(
                    semantica,
                    start=1
                )
            }

            posicoes_lexicais = {
                int(indice): posicao
                for posicao, indice in enumerate(
                    lexical,
                    start=1
                )
            }

            rankings.append((
                consulta,
                posicoes_semanticas,
                posicoes_lexicais
            ))

        posicoes_fusao = {
            indice: posicao
            for posicao, indice in enumerate(
                candidatos,
                start=1
            )
        }

        print(
            f"Arquivo identificado: {arquivo_alvo}"
        )

        for indice in indices_teste:
            print(
                f"\nTrecho de índice {indice}"
            )

            print(
                f"Arquivo: {self.arquivos[indice]}"
            )

            print(
                f"Página: "
                f"{self.trechos[indice]['pagina']}"
            )

            posicao_final = (
                posicoes_fusao.get(indice)
            )

            if posicao_final is not None:
                print(
                    f"Posição na fusão: "
                    f"{posicao_final}"
                )

                print(
                    f"Pontuação final: "
                    f"{pontuacoes_fusao[indice]:.6f}"
                )
            else:
                print(
                    "Trecho ausente da fusão."
                )

            for (
                consulta,
                posicoes_semanticas,
                posicoes_lexicais
            ) in rankings:
                posicao_semantica = (
                    posicoes_semanticas.get(
                        indice,
                        "Fora dos 100 primeiros"
                    )
                )

                posicao_lexical = (
                    posicoes_lexicais.get(
                        indice,
                        "Fora dos 100 primeiros"
                    )
                )

                print(
                    f"\nConsulta: {consulta}"
                )

                print(
                    f"Ranking semântico: "
                    f"{posicao_semantica}"
                )

                print(
                    f"Ranking BM25: "
                    f"{posicao_lexical}"
                )

        print(
            "\n=== FIM DO DIAGNÓSTICO ===\n"
        )

    def buscar(self, pergunta, limite=3):
        if not self.trechos or limite <= 0:
            return []

        arquivo_alvo = self.identificar_arquivo(
            pergunta
        )

        termos_foco = None

        if arquivo_alvo:
            nome = (
                arquivo_alvo.rsplit(".", 1)[0]
                .replace("_", " ")
                .replace("-", " ")
            )

            termos_foco = (
                extrair_raizes(pergunta)
                - extrair_raizes(nome)
            )

        consultas = (
            gerar_consultas(pergunta)
            or [pergunta]
        )

        print(
            "\n=== CONSULTAS GERADAS ==="
        )

        for numero, consulta in enumerate(
            consultas,
            start=1
        ):
            print(
                f"{numero}. {consulta}"
            )

        print(
            "========================\n"
        )

        pontuacoes_fusao = {}

        for posicao_consulta, consulta in enumerate(
            consultas
        ):
            semantica, lexical = (
                self.classificar_consulta(
                    consulta,
                    arquivo=arquivo_alvo,
                    termos_foco=termos_foco
                )
            )

            peso_consulta = (
                2.0
                if posicao_consulta == 0
                else 1.0
            )

            for classificacao in (
                semantica,
                lexical
            ):
                for posicao, indice in enumerate(
                    classificacao
                ):
                    indice = int(indice)

                    pontuacoes_fusao[indice] = (
                        pontuacoes_fusao.get(
                            indice,
                            0.0
                        )
                        + peso_consulta
                        / (60 + posicao + 1)
                    )

        candidatos = sorted(
            pontuacoes_fusao,
            key=pontuacoes_fusao.get,
            reverse=True
        )

        # Diagnóstico temporário.
        self.diagnosticar_ranking(
            consultas,
            arquivo_alvo,
            termos_foco,
            candidatos,
            pontuacoes_fusao
        )

        resultados = []
        selecionados = []

        for indice in candidatos:
            if len(resultados) >= limite:
                break

            arquivo = (
                self.arquivos[indice]
                or "Documento desconhecido"
            )

            if any(
                abs(indice - anterior) <= 1
                and self.arquivos[anterior] == arquivo
                for anterior in selecionados
            ):
                continue

            selecionados.append(indice)

            inicio = indice

            if (
                indice > 0
                and self.arquivos[indice - 1] == arquivo
            ):
                inicio = indice - 1

            fim = indice + 1

            if (
                indice + 1 < len(self.trechos)
                and self.arquivos[indice + 1] == arquivo
            ):
                fim = indice + 2

            partes = []
            paginas = set()

            for vizinho in range(inicio, fim):
                trecho = self.trechos[vizinho]

                paginas.add(
                    trecho["pagina"]
                )

                partes.append(
                    f"Documento: {arquivo}\n"
                    f"Página {trecho['pagina']}:\n"
                    f"{trecho['texto']}"
                )

            resultados.append({
                "arquivo": arquivo,
                "pagina": self.trechos[indice]["pagina"],
                "paginas": sorted(paginas),
                "texto": "\n\n".join(partes),
                "pontuacao": float(
                    pontuacoes_fusao[indice]
                )
            })

        return resultados
