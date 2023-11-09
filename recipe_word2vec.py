import os
import sys
import logging
import unidecode
import ast

import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

import config
from finding_scraps import *



def get_and_sort_corpus(data):
    corpus_sorted = []

    for doc in data.parsed.values:
        doc.sort()
        corpus_sorted.append(doc)
    
    return corpus_sorted

def get_recommendations(N, scores):
    df_recipes = pd.read_csv(config.PARSED_PATH)
    top = sorted(range(len(scores)), key = lambda i: scores[i], reverse = True)[:N]
    recommendation = pd.DataFrame(columns = ['recipe', 'ingredients', 'score', 'url'])
    count = 0
    for i in top:
        recommendation.at[count, 'recipe'] = title_parser(df_recipes['name'][i])
        recommendation.at[count, 'ingredients'] = ingredient_parser_final(
            df_recipes['ingredient'][i]
        )
        recommendation.at[count, 'url'] = df_recipes['link'][i]
        recommendation.at[count, 'score'] = f"{scores[i]}"
        count += 1
    return recommendation

def title_parser(title):
    title = unidecode.unidecode(title)
    return title

def ingredient_parser_final(ingredient):

    if isinstance(ingredient, list):
        ingredients = ingredient
    else:
        ingredient = strToList(ingredient)
        ingredients = ast.literal_eval(ingredient)

    ingredients = ",".join(ingredients)
    ingredients = unidecode.unidecode(ingredients)

    return ingredients

class MeanEmbeddingVectorizer(object):
    def __init__(self, word_model):
        self.word_model = word_model
        self.vector_size = word_model.wv.vector_size

    def fit(self):
        return self
    
    def transform(self, docs):
        doc_word_vector = self.word_average_list(docs)
        return doc_word_vector
    
    def word_average(self, sent):
        mean = []

        for word in sent:
            if word in self.word_model.wv.index_to_key:
                mean.append(self.word_model.wv.get_vector(word))

        if not mean:
            return np.zeros(self.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean
        
    def word_average_list(self, docs):
        return np.vstack([self.word_average(sent) for sent in docs])
    
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word_model):
        self.word_model = word_model
        self.word_idf_weight = None
        self.vector_size = word_model.wv.vector_size

    def fit(self, docs):
        text_docs = []
        for doc in docs:
            text_docs.append(" ".join(doc))

        tfidf = TfidfVectorizer()
        tfidf.fit(text_docs)

        max_idf = max(tfidf.idf_)
        self.word_idf_weight = defaultdict(
            lambda: max_idf,
            [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()],
        )
        return self
    
    def transform(self, docs):
        doc_word_vector = self.word_average_list(docs)
        return doc_word_vector
    
    def word_average(self, sent):
        mean = []
        for word in sent:
            if word in self.word_model.wv.index_to_key:
                mean.append(
                    self.word_model.wv.get_vector(word) * self.word_idf_weight[word]
                )

        if not mean:
            return np.zeros(self.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean
        
    def word_average_list(self, docs):
        return np.vstack([self.word_average(sent) for sent in docs])
    
def get_recs(ingredients, N=5, mean=False):
    model = Word2Vec.load('models/model_cbow.bin')
    model.init_sims(replace=True)
    if model:
        print('Successfully loaded model')
    data = pd.read_csv('df_parsed.csv')
    data['parsed'] = data.ingredient.apply(ingredient_parser)
    corpus = get_and_sort_corpus(data)

    if mean:
        mean_vec_tr = MeanEmbeddingVectorizer(model)
        doc_vec = mean_vec_tr.transform(corpus)
        doc_vec = [doc.reshape(1, -1) for doc in doc_vec]
        assert len(doc_vec) == len(corpus)
    else:
        tfidf_vec_tr = TfidfEmbeddingVectorizer(model)
        tfidf_vec_tr.fit(corpus)
        doc_vec = tfidf_vec_tr.transform(corpus)
        doc_vec = [doc.reshape(1, -1) for doc in doc_vec]
        assert len(doc_vec) == len(corpus)

    input = ingredients
    input = input.split(",")
    input = ingredient_parser(input)

    if mean:
        input_embedding = mean_vec_tr.transform([input])[0].reshape(1, -1)
    else:
        input_embedding = tfidf_vec_tr.transform([input])[0].reshape(1, -1)

    cos_sim = map(lambda x: cosine_similarity(input_embedding, x)[0][0], doc_vec)
    scores = list(cos_sim)
    recommendations = get_recommendations(N, scores)
    return recommendations
    
if __name__ == '__main__':
    input = 'chicken thigh, risdlfgbviahsddsagv, onion, rice noodle, seaweed nori sheet, sesame, shallot, soy, spinach, star, tofu'
    rec = get_recs(input)
    print(rec)