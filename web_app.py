# Imports
import streamlit as st
import pickle
import pandas as pd
import numpy as np

import model_helper_functions

# Constants
MODEL_PATH = './'
MODEL_FILE_NAME = 'rf_tfidf_plus_guardian_model.sav'
RANDOM_STATE = 42
DATA_PATH = './'

# Local Model Helper Function and Stopwords list
gist_file = open(DATA_PATH + "gist_stopwords.txt", "r")
try:
    content = gist_file.read()
    expanded_stopwords = content.split(",")
finally:
    gist_file.close()

expanded_stopwords.remove('via')
expanded_stopwords.remove('eu')
expanded_stopwords.remove('uk')

def lowercase_and_only_expanded_stopwords(doc):
    """Remove stopwords and lowercase tokens"""
    stop_words = expanded_stopwords
    return [token.lower() for token in doc if token.lower() in stop_words]

# Load pipeline
@st.cache(allow_output_mutation=True)
def load_pipeline(model_path=MODEL_PATH, model_file_name=MODEL_FILE_NAME):
    """
    Load the Text Processing and Classifier Pipeline
    """
    return pickle.load(open(model_path + model_file_name, 'rb'))

pipeline = load_pipeline()


st.title('News Classification')

st.write("""
        Enter the title and text of a news story and a trained random forest
        classifier will classify it as Truthful or Fake. Please note that the
        algorithm is not checking the facts of the news story, it is basing
        the classification on the style of the text of the story; specifically, it
        is basing the classification only on the stop words (common words) in
        the story and its title.
         """)

news_title = st.text_input('Enter a News Title')

if news_title:
    news_story = st.text_area('Enter a News Story', height=400)

    if news_story and news_title:
        tokens = model_helper_functions.tokenize_and_normalize_title_and_text(news_title, news_story)
        stop_words_only = lowercase_and_only_expanded_stopwords(tokens)
        if len(stop_words_only) == 0:
            st.write('There were no stopwords in your news title and story.')
        else:
            class_ = pipeline.predict([tokens])
            if class_ == 0:
                class_text = 'Fake'
            else:
                class_text = 'Truthful'

            probability = round(pipeline.predict_proba([tokens])[0][class_][0] * 100, 2)
            st.subheader('Classification Results')
            st.write('Your news story is classified as ', class_text, 'with a ',
                     probability, '% probability.')
            st.write()
            st.subheader('Your news story with only stop words:')
            st.write(' '.join(stop_words_only))
