def gerar_resposta(pergunta, resultados):
    if not resultados:
        return (
            "Não encontrei informações suficientes "
            "no livro para responder."
        )

    resposta = f"Pergunta: {pergunta}\n\n"
    resposta += "Informações encontradas no livro:\n"

    for resultado in resultados:
        pagina = resultado["pagina"]
        texto = resultado["texto"]

        resposta += f"\nPágina {pagina}:\n"
        resposta += f"{texto[:500]}\n"

    return resposta