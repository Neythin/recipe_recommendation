import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from finding_scraps import *

import pickle
import config
import unidecode
import ast

# Get N recommendations based on score
def get_recommendations(N, scores):
    df_recipes = pd.read_csv(config.PARSED_PATH)
    
    # For a list of scores, sort in descending order and assign to variable "top"
    top = sorted(range(len(scores)), key = lambda i: scores[i], reverse = True)[:N]
    # Create a DataFrame containing the recipe name, ingredients, score, and url
    recommendation = pd.DataFrame(columns = ['recipe', 'ingredients', 'score', 'url'])
    # Initialize a count variable
    count = 0

    for i in top:
        # Extracts the recipe name and stores it in the recipe column of the new dataframe
        recommendation.at[count, 'recipe'] = title_parser(df_recipes['name'][i])
        # Extracts and parses the ingredients, storing the result in the ingredient column
        recommendation.at[count, 'ingredients'] = ingredient_parser_final(
            df_recipes['ingredient'][i]
        )
        # Retrieves the recipe URL from the dataframe and stores it
        recommendation.at[count, 'url'] = df_recipes['link'][i]
        # Converts the similarity score to a string with three decimal places and stores it in the score column
        recommendation.at[count, 'score'] = '{:.3f}'.format(float(scores[i]))
        count += 1

    return recommendation

def ingredient_parser_final(ingredient):

    # If ingredients are in the format of a list, assign it to the ingredients variable
    if isinstance(ingredient, list):
        ingredients = ingredient
    # If not, convert the string to a list
    else:
        ingredient = strToList(ingredient)
        ingredients = ast.literal_eval(ingredient)

    # Join the list of ingredients into a single string, separate with commas
    ingredients = ",".join(ingredients)
    # Get rid of accents
    ingredients = unidecode.unidecode(ingredients)

    return ingredients

def title_parser(title):
    # Get rid of accents in the title
    title = unidecode.unidecode(title)
    return title

def RecSys(ingredients, N=5):
    
    # Opens two files containing the TF-IDF encodings and the TF-IDF model
    with open(config.TFIDF_ENCODING_PATH, 'rb') as f:
        tfidf_encodings = pickle.load(f)

    with open(config.TFIDF_MODEL_PATH, 'rb') as f:
        tfidf = pickle.load(f)

    # Tries to parse the input ingredients, if there is an exception, catch it and parse a list containing the input ingredients as a single element
    try:
        ingredients_parsed = ingredient_parser(ingredients)
    except:
        ingredients_parsed = ingredient_parser([ingredients])

    # Join the parsed ingredients into a single string
    ingredients_parsed = " ".join(ingredients_parsed)
    # Applies the pre-trained TF-IDF model to transform the joined ingredients into a TF-IDF representation
    ingredients_tfidf = tfidf.transform([ingredients_parsed])

    # Calculates the cosine similarity between the TF-IDF represenation of the input ingredients and the TF-IDF encodings of a set of recipes
    cos_sim = map(lambda x: cosine_similarity(ingredients_tfidf, x), tfidf_encodings)
    # Converts the resulting map object to a list of cosine similarity scores
    scores = list(cos_sim)

    # Get the list of recommendations
    recommendations = get_recommendations(N, scores)
    return recommendations

if __name__ == "__main__":
    # test ingredients
    test_ingredients = "pasta, tomato, onion"
    recs = RecSys(test_ingredients)
    print(recs.score)