from sklearn.metrics import classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def passthrough(doc):
    """passthrough function for use in the pipeline because the text is already tokenized"""
    return doc

def confustion_matrix_and_classification_report(estimator, X, y, labels, set_name):
    """
    Display a Classfication Report and Confusion Matrix for the given data.
    """

    predictions = estimator.predict(X)

    print(f'Classification Report for {set_name} Set')
    print(classification_report(y, predictions, target_names=labels))

    matrix = plot_confusion_matrix(estimator,
                                   X,
                                   y,
                                   display_labels = labels,
                                   cmap = plt.cm.Blues,
                                   xticks_rotation = 70,
                                   values_format = 'd')
    matrix.ax_.set_title(f'{set_name} Set Confustion Matrix, without Normalization')

    plt.show()

    matrix = plot_confusion_matrix(estimator,
                                   X,
                                   y,
                                   display_labels = labels,
                                   cmap = plt.cm.Blues,
                                   xticks_rotation = 70,
                                   normalize = 'true')
    matrix.ax_.set_title(f'{set_name} Set Confustion Matrix, with Normalization')

    plt.show()

class LemmaTokenizer:
    def __init__(self):
         self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in doc]

def remove_stopwords(doc):
    """Remove the stopwords from the input document"""
    stop_words = stopwords.words('english')
    return [token for token in doc if ((token not in stop_words) and (token.lower() not in stop_words))]

def lowercase_tokens(doc):
    """lowercase all letters in doc"""
    return [token.lower() for token in doc]

def lowercase_and_remove_stopwords(doc):
    """Remove stopwords and lowercase tokens"""
    stop_words = stopwords.words('english')
    return [token.lower() for token in doc if token.lower() not in stop_words]
