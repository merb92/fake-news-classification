# mod5
import os
import zipfile
import pandas as pd
import numpy as np
import re
from re import search
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = '../data/'
# if the path or names of the data hosted on Kaggle changes the following
# four constants will need updated
FAKE_TRUE_DATA_ARCHIVE = 'fake-and-real-news-dataset.zip'
FAKE_DATA_FILE = 'Fake.csv'
TRUE_DATA_FILE = 'True.csv'
FAKE_TRUE_KAGGLE_LOCATION = 'clmentbisaillon/fake-and-real-news-dataset'

GUARDIAN_DATA_ARCHIVE = 'guardian-news-dataset.zip'
GUARDIAN_DATA_FILE = 'combined_filtered.csv'
GUARDIAN_KAGGLE_DATA_LOCATION = 'sameedhayat/guardian-news-dataset'

RANDOM_STATE = 42

def remove_single_characters(word_list, exception_list):
    """Remove all the single characters, except those on the exception list"""
    return [w for w in word_list if (len(w) > 1 or w in exception_list)]

def remove_words(word_list, words_to_remove):
    """Remove all the words in the words_to_remove list from the words_list"""
    return [w for w in word_list if w not in words_to_remove]

def lower_unless_all_caps(string_):
    """
    Make all words in the input string lowercase unless that
    word is in all caps
    """
    words = string_.split()
    processed_words = [w.lower() if not (w.isupper() and len(w) > 1) else w for w in words]
    return ' '.join(processed_words)

def tokenize_and_normalize_title_and_text(title, text):
    """Combine, tokenize, and normalize the title and text of a news story"""

    URL_REGEX = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    TWITTER_HANDLE_REGEX = r'(?<=^|(?<=[^\w]))(@\w{1,15})\b'
    DATE_WORDS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday',
              'saturday', 'sunday', 'january', 'february', 'march', 'april',
             'may', 'june', 'july', 'august', 'september', 'october',
             'november', 'december']

    title_text = ' '.join([title, text])
    title_text = re.sub(URL_REGEX, '{link}', title_text)
    title_text = re.sub(TWITTER_HANDLE_REGEX, '@twitter-handle', title_text)
    title_text = lower_unless_all_caps(title_text)
    title_text = re.sub(r'\d+', ' ', title_text)
    title_text = re.sub(r'\(reuters\)', ' ', title_text)
    tokens = word_tokenize(title_text)
    tokens = remove_single_characters(tokens, ['i', '!'])
    tokens = remove_words(tokens, ["'s"])
    tokens = remove_words(tokens, DATE_WORDS)

    return tokens

def not_a_date(string):
    """
    Check if the input string can be converted to a date.

    Return:
    True if it is not a date
    False if it is a date
    """
    try:
        date = pd.to_datetime(string)
        return False
    except ValueError:
        return True

def lowercase_and_only_expanded_stopwords(doc):
    """Remove stopwords and lowercase tokens"""

    # Get list of common words (stop words)

    # using expanded stopwords list from https://gist.github.com/sebleier/554280

    gist_file = open(DATA_PATH + "gist_stopwords.txt", "r")
    try:
        content = gist_file.read()
        expanded_stopwords = content.split(",")
    finally:
        gist_file.close()

    # Remove some words that are particular to the dataset

    expanded_stopwords.remove('via')
    expanded_stopwords.remove('eu')
    expanded_stopwords.remove('uk')

    return [token.lower() for token in doc if token.lower() in expanded_stopwords]

def passthrough(doc):
    """passthrough function for use in the pipeline because the text is already tokenized"""
    return doc

# def download_data():
#     """
#     Download data from Kaggle and save in DATA_PATH directory
#     """
#
#     try:
#         os.mkdir(DATA_PATH)
#     except FileExistsError:
#         print('Data directory already exists')
#         pass
#
#     # Download Fake and True News Dataset
#
#     # assumes that kaggle has been installed and an api key is correctly installed
#     if not os.path.exists(DATA_PATH + FAKE_TRUE_DATA_ARCHIVE):
#         !kaggle datasets download -d $FAKE_TRUE_KAGGLE_LOCATION -p $DATA_PATH
#
#     if not os.path.exists(DATA_PATH + FAKE_DATA_FILE):
#         with zipfile.ZipFile(DATA_PATH + DATA_ARCHIVE, 'r') as zip_ref:
#             zip_ref.extractall(path=DATA_PATH)
#
#     # Download Guardian Dataset
#
#     # assumes that kaggle has been installed and an api key is correctly installed
#     if not os.path.exists(DATA_PATH + GUARDIAN_DATA_ARCHIVE):
#         !kaggle datasets download -d $GUARDIAN_KAGGLE_DATA_LOCATION -p $DATA_PATH
#
#     if not os.path.exists(DATA_PATH + GUARDIAN_DATA_FILE):
#         with zipfile.ZipFile(DATA_PATH + GUARDIAN_DATA_ARCHIVE, 'r') as zip_ref:
#             zip_ref.extractall(path=DATA_PATH)

def load_data():
    """
    Load data stored in .csv files into dataframes
    """

    fake_df = pd.read_csv(DATA_PATH + FAKE_DATA_FILE)
    true_df = pd.read_csv(DATA_PATH + TRUE_DATA_FILE)
    guard_df = pd.read_csv(DATA_PATH + GUARDIAN_DATA_FILE)

    # add labels to data frames. All the Guardian news stories are assumed to be
    # true because The Guardian is not designated as a "Fake" news source.

    fake_df['label'] = 'fake'
    true_df['label'] = 'true'
    guard_df['label'] = 'true'

    # Filter the Guardian dataset to only contain "Politics" stories

    guard_politics_df = guard_df[guard_df.sectionName == 'Politics']

    # Remove unneeded columns from Guardian Dataset

    guard_clean_df = guard_politics_df[['fields.bodyText', 'webTitle', 'label']]

    # Update columns in guard_clean_df to match fake and true dfs

    guard_clean_df.columns = ['text', 'title', 'label']

    # Combine fake and true clean dfs for futher cleaning in case there are any
    # stories that are labeled Fake and True

    fake_true_clean_df = pd.concat([fake_df, true_df], ignore_index=True)

    return fake_true_clean_df, guard_clean_df

def clean_data(fake_true_clean_df, guard_clean_df):
    """
    Clean the input dataframes
    """
    # Remove Duplicates

    fake_true_clean_df = fake_true_clean_df.drop_duplicates(ignore_index=True)
    guard_clean_df = guard_clean_df.drop_duplicates(ignore_index=True)

    # Remove rows with Nan, first assigning spaces to Nan

    fake_true_clean_df = fake_true_clean_df.replace(r'^\s*$', np.nan, regex=True)
    guard_clean_df = guard_clean_df.replace(r'^\s*$', np.nan, regex=True)

    fake_true_clean_df.dropna(inplace=True)
    guard_clean_df.dropna(inplace=True)

    # Drop Duplicate News Stories.  Duplicate Titles are OK

    fake_true_clean_df = fake_true_clean_df.drop_duplicates(['text'], ignore_index=True)
    guard_clean_df = guard_clean_df.drop_duplicates(['text'], ignore_index=True)

    # Check the date column in fake_true_clean_df to find malformed data and remove it

    bad_date_indexes = fake_true_clean_df[fake_true_clean_df['date'].apply(not_a_date)].index

    fake_true_clean_df = fake_true_clean_df.drop(bad_date_indexes, axis=0)

    return fake_true_clean_df, guard_clean_df

def normalize_and_tokenize(fake_true_clean_df, guard_clean_df):
    """
    Normalize the title and text and the tokenize the combined columns
    """

    fake_true_clean_df['tt_tokens'] = fake_true_clean_df.apply(lambda row: tokenize_and_normalize_title_and_text(row['title'], row['text']), axis=1)

    guard_clean_df['tt_tokens'] = guard_clean_df.apply(lambda row: tokenize_and_normalize_title_and_text(row['title'], row['text']), axis=1)

    return fake_true_clean_df, guard_clean_df

def integrate_datasets(fake_true_clean_df, guard_clean_df):
    """
    Take a random sample of the true news stories from fake_true_clean_df that is 1/2
    the size of the fake stories. Take a random sample that is 1/2
    the size of the fake stories of the stories in guard_clean_df
    which are also labeled true.  Join these two samples with all the fake News
    stories.
    """

    df_fake = fake_true_clean_df[fake_true_clean_df.label == 'fake']
    df_true = fake_true_clean_df[fake_true_clean_df.label == 'true']

    true_sample_size = int(len(df_fake) / 2)

    df_true_sample = df_true.sample(true_sample_size, random_state=RANDOM_STATE)
    df_guard_sample = guard_clean_df.sample(true_sample_size, random_state=RANDOM_STATE)

    df_all = pd.concat([df_fake[['tt_tokens', 'label']],
                        df_true_sample[['tt_tokens', 'label']],
                        df_guard_sample[['tt_tokens', 'label']]],
                       axis=0)

    return df_all

def train_model(df):
    """
    Train a Random Forest Classifier on the input news dataset using tf/idf to
    represent the text.  Only words from a set list of common words will be used
    in the model.
    """

    # Split the data
    y = df.label
    X = df.drop('label', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=RANDOM_STATE,
                                                        stratify=y)

    # Encode labels

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)



    # Define pipeline for the model

    pipeline = Pipeline([('bow',CountVectorizer(#min_df = 5,
                                                preprocessor = lowercase_and_only_expanded_stopwords,
                                                tokenizer = passthrough,
                                                #max_df = 1.0,
                                                ngram_range = (1, 1))),
                         ('tfidf', TfidfTransformer()),
                         ('classifier', RandomForestClassifier(n_jobs = -1,
                                                               random_state = RANDOM_STATE,
                                                               min_samples_split = 0.005,
                                                               max_depth = None)),
                        ])

    # Fit the model

    pipeline.fit(X_train, y_train_enc)

    # Display classification reports for the test sets

    y_hat_test = pipeline.predict(X_test)

    print(f'Classification Report for the Test Set')
    print(classification_report(y_test_enc, y_hat_test, target_names=le.classes_))

    matrix = plot_confusion_matrix(pipeline,
                                   X_test,
                                   y_test_enc,
                                   display_labels = le.classes_,
                                   cmap = plt.cm.Blues,
                                   xticks_rotation = 70,
                                   values_format = 'd')

    matrix.ax_.set_title(f'Test Set Confustion Matrix, without Normalization')

    plt.show()

def main():
    # download_data()
    fake_true_clean_df, guard_clean_df = load_data()
    fake_true_clean_df, guard_clean_df = clean_data(fake_true_clean_df, guard_clean_df)
    fake_true_clean_df, guard_clean_df = normalize_and_tokenize(fake_true_clean_df, guard_clean_df)
    df_all = integrate_datasets(fake_true_clean_df, guard_clean_df)
    print(len(df_all))
    train_model(df_all)


if __name__ == "__main__":
    main()
