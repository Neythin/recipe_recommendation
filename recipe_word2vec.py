from gensim.models import Word2Vec

def get_and_sort_corpus(data):
    corpus_sorted = []
    for doc in data.parsed.values:
        doc.sort()
        corpus_sorted.append(doc)
    return corpus_sorted

def get_window(corpus):
    lengths = [len(doc) for doc in corpus]
    avg_len = float(sum(lengths))/len(lengths)
    return round(avg_len)