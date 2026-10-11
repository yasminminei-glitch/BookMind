
def dividir_texto(texto, pagina, tamanho=250, sobreposicao=50):
    """
    Divide o texto em trechos com sobreposição.

    Mantém informações de contexto entre trechos
    consecutivos e preserva a página de origem.
    """

    if tamanho <= 0:
        raise ValueError("O tamanho deve ser maior que zero.")

    if not 0 <= sobreposicao < tamanho:
        raise ValueError(
            "A sobreposição deve ser menor que o tamanho."
        )

    trechos = []
    palavras = texto.split()

    if not palavras:
        return trechos

    passo = tamanho - sobreposicao

    for inicio in range(0, len(palavras), passo):
        grupo = palavras[inicio:inicio + tamanho]

        if not grupo:
            continue

        trecho = {
            "pagina": pagina,
            "texto": " ".join(grupo)
        }

        trechos.append(trecho)

        if inicio + tamanho >= len(palavras):
            break

    return trechos
