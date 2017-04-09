def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6
