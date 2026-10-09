
def dividir_texto(texto, pagina, tamanho=150):
    trechos = []
    palavras = texto.split()

    for inicio in range(0, len(palavras), tamanho):
        grupo = palavras[inicio:inicio + tamanho]

        trecho = {
            "pagina": pagina,
            "texto": " ".join(grupo)
        }

        trechos.append(trecho)

    return trechos
