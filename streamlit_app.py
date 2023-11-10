import streamlit as st
import pandas as pd
import numpy as np

import os
import config
import nltk
from finding_scraps import *
from rec_sys import *
from recipe_word2vec import *

try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')



def make_clickable(name, link):
    text = name
    return f'<a target="_blank" href="{link}">{text}</a>'



def main():
    st.title("Scraps.")

    """
    ingredients = st.text_input(
        "Enter your ingredients:",
        "Milk, eggs, cheese"
    )

    execute_recsys = st.button("Make a meal")

    if execute_recsys:

        recipe = get_recs(ingredients, mean=True)
        recipe["url"] = recipe.apply(
            lambda row: make_clickable(row["recipe"], row["url"], axis=1)
        )
        recipe_display = recipe[["recipe", "url", "ingredients"]]
    """

    st.text_input(
        "Enter your ingredients:",
        "Milk, eggs, cheese"
    )
    st.button("Make a meal")
    



if __name__ == "__main__":
    main()