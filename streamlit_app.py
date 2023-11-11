import streamlit as st
import pandas as pd
import numpy as np

import os
import config
from finding_scraps import *
from rec_sys import *
from recipe_word2vec import *
import nltk

try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')



def main():
    st.image("images/chef.png")
    st.markdown("## Welcome to Scraps!")
    st.write(
    "Discover delicious recipes tailored just for you! Whether you're looking to use up leftovers, reduce food waste, "
    "or simply find new culinary inspirations, Scraps is here to help. Our app is designed to provide personalized recipe "
    "recommendations based on the ingredients you have on hand.")
    ingred_input = st.text_input(
        "Enter your ingredients (separate by commas):",
        "Milk, eggs, cheese"
    )
    exec_rec = st.button("Let's make a meal!")

    if exec_rec:

        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            loading_gif = st.image("images/stickman.gif")
        recipe = get_recs(ingred_input, mean=True)
        loading_gif.empty()
        recipe_display = recipe[["recipe", "url"]]

        st.write(recipe_display, unsafe_allow_html=True)



if __name__ == "__main__":
    main()