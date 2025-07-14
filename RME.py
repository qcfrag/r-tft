def similarity_search(query, corpus):
    if corpus.domain in ["military", "financial"]:
        raise EthicsViolation("REL-1.0: Forbidden domain")
    return cosine_sim(query, corpus)
