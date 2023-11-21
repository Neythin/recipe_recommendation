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
from rec_sys import *



# Sort a corpus of documents in alphabetical order, so that ingredients put in a different order don't have different contexts
def get_and_sort_corpus(data):
    # Assign an empty list to be filled with sorted documents
    corpus_sorted = []

    # for each word in the parsed items column of the data set, sort the words by ascending order and add it to the new list
    for doc in data.parsed.values:
        doc.sort()
        corpus_sorted.append(doc)
    
    return corpus_sorted



class MeanEmbeddingVectorizer(object):
    def __init__(self, word_model):
        self.word_model = word_model
        self.vector_size = word_model.wv.vector_size

    # Define a function to reference to the instance of the class, helps with method chaining
    def fit(self):
        return self
    
    # Assign a variable that contains a 2D array where each row represents the mean vector of a document
    def transform(self, docs):
        doc_word_vector = self.word_average_list(docs)
        return doc_word_vector
    
    # Define a function to return a single vector representing the average of all word vectors
    def word_average(self, sent):
        # Initialize empty list to store word vectors
        mean = []

        # For each word in a list of words... 
        for word in sent:
            # If the word exists in the vocabulary of the word model, add the word vector to the mean list
            if word in self.word_model.wv.index_to_key:
                mean.append(self.word_model.wv.get_vector(word))

        # If the mean list is empty, return an array of zeros with a size equal to the vector size
        if not mean:
            return np.zeros(self.vector_size)
        # If the mean list is not empty, convert the list to an array and calculate the mean along the columns
        else:
            mean = np.array(mean).mean(axis=0)
            return mean
        
    # Self = a reference to the instance of the class; docs = a list of lists, where each inner list represents a document of words
    def word_average_list(self, docs):
        # With a list of documents, calculate the word average for each document, return a list of mean vectors, and return them in a 2D array
        return np.vstack([self.word_average(sent) for sent in docs])
    


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word_model):
        self.word_model = word_model
        self.word_idf_weight = None
        self.vector_size = word_model.wv.vector_size

    def fit(self, docs):
        # Initialize an empty list to add all the documents
        text_docs = []
        for doc in docs:
            text_docs.append(" ".join(doc))

        # Initialize a vectorizer that converts text documents into a TF-IDF representation (Term Frequency = Inverse Document Frequency)
        tfidf = TfidfVectorizer()
        # Fit determines the vocabulary and computes the IDF weights
        tfidf.fit(text_docs)

        # Calculates the max IDF value from the model
        max_idf = max(tfidf.idf_)
        # Creates a dictionary using defaultdict and sets max_idf to default value, the dictionary pairs each word with its IDF weight
        self.word_idf_weight = defaultdict(
            lambda: max_idf,
            [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()],
        )
        return self
    
    # Assign a variable that contains a 2D array where each row represents the mean vector of a document
    def transform(self, docs):
        doc_word_vector = self.word_average_list(docs)
        return doc_word_vector
    
    # Define a function to return a single vector representing the average of all word vectors
    def word_average(self, sent):
        # Initialize empty list to store word vectors
        mean = []

        # For each word in a list of words...
        for word in sent:
            # If the word exists in the vocabulary of the word model, scale the word vector by its IDF weight and append to list
            if word in self.word_model.wv.index_to_key:
                mean.append(
                    self.word_model.wv.get_vector(word) * self.word_idf_weight[word]
                )

        # If the mean list is empty, return an array of zeros with a size equal to the vector size
        if not mean:
            return np.zeros(self.vector_size)
        # If the mean list is not empty, convert the list to an array and calculate the mean along the columns
        else:
            mean = np.array(mean).mean(axis=0)
            return mean

    # Self = a reference to the instance of the class; docs = a list of lists, where each inner list represents a document of words    
    def word_average_list(self, docs):
        # With a list of documents, calculate the word average for each document, return a list of mean vectors, and return them in a 2D array
        return np.vstack([self.word_average(sent) for sent in docs])
    


def get_recs(ingredients, N=5, mean=False):
    # Load a pre-trained Word2Vec model
    model = Word2Vec.load('models/model_cbow.bin')
    # Initializes the model, which includes normalizing vectors, making subsequent operations more memory-efficient
    model.init_sims(replace=True)
    if model:
        print('Successfully loaded model')
    data = pd.read_csv('df_parsed.csv')
    # Parse the ingredients and create new column for it
    data['parsed'] = data.ingredient.apply(ingredient_parser)
    # Sorts a corpus from the parsed ingredients data frame
    corpus = get_and_sort_corpus(data)

    # Depending on the mean, the function chooses between creating document vectors using either mean embedding or TF-IDF embedding
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

    # The input ingredients are processed by splitting them into a list, parsing them, and embedding them using either mean or TF=IDF embedding
    input = ingredients
    input = input.split(",")
    input = ingredient_parser(input)

    if mean:
        input_embedding = mean_vec_tr.transform([input])[0].reshape(1, -1)
    else:
        input_embedding = tfidf_vec_tr.transform([input])[0].reshape(1, -1)

    # Calculates the cosine similarity between the input embedding and each document vector in the corpus
    cos_sim = map(lambda x: cosine_similarity(input_embedding, x)[0][0], doc_vec)
    # Converts the cosine similarity score to a list
    scores = list(cos_sim)
    # Returns the list of recommendations
    recommendations = get_recommendations(N, scores)
    return recommendations
    


if __name__ == '__main__':
    input = 'chicken thigh, risdlfgbviahsddsagv, onion, rice noodle, seaweed nori sheet, sesame, shallot, soy, spinach, star, tofu'
    rec = get_recs(input)
    print(rec)