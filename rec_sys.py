import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from finding_scraps import *

import pickle
import config
import unidecode
import ast

def get_recommendations(N, scores):
    df_recipes = pd.read_csv(config.PARSED_PATH)
    top = sorted(range(len(scores)), key = lambda i: scores[i], reverse = True)[:N]
    recommendation = pd.DataFrame(columns = ['recipe', 'ingredients', 'score', 'url'])
    count = 0