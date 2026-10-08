import re
import unicodedata


def normalizar(texto):
    texto = texto.lower()

    texto = unicodedata.normalize("NFD", texto)

    texto = "".join(
        caractere
        for caractere in texto
        if unicodedata.category(caractere) != "Mn"
    )

    return texto


def buscar_trechos(pergunta, trechos, limite=5):
    palavras = re.findall(r"\w+", normalizar(pergunta))

    ignorar = {"quem", "que", "qual", "como", "onde", "quando",
               "por", "para", "com", "uma", "uns", "das", "dos",
               "era", "foi", "sao", "sobre"}

    palavras = [
        palavra for palavra in palavras
        if palavra not in ignorar and len(palavra) > 2
    ]

    resultados = []

    for trecho in trechos:
        texto_normalizado = normalizar(trecho["texto"])

        pontuacao = sum(
            len(re.findall(r"\b" + re.escape(palavra) + r"\b", texto_normalizado))
            for palavra in palavras
        )

        if pontuacao > 0:
            resultados.append({
                "pagina": trecho["pagina"],
                "texto": trecho["texto"],
                "pontuacao": pontuacao
            })

    resultados.sort(
        key=lambda resultado: resultado["pontuacao"],
        reverse=True
    )

    return resultados[:limite]