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



def make_clickable(name, link):
    text = name
    return f'<a target="_blank" href="{link}">{text}</a>'



def main():
    st.title("Scraps.")
    st.markdown("")
    st.text_input(
        "Enter your ingredients:",
        "Milk, eggs, cheese"
    )
    st.button("Make a meal")



if __name__ == "__main__":
    main()