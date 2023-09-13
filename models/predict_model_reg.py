import argparse
import pandas as pd
import joblib

import pandas as pd
pd.set_option('display.max_colwidth', 280)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
from tqdm.notebook import tqdm_notebook as tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, auc
from sklearn.metrics import f1_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
stop_words = set(stopwords.words('russian'))

RANDOM_STATE = 1052023
import warnings
warnings.filterwarnings('ignore')

def text_preprocessing(text):
    """
    Выполняет предобработку входного текста, включая удаление стоп-слов, приведение к нижнему регистру,
    лемматизацию и стемминг.

    Параметры:
        text (str): Входной текст для предобработки.

    Возвращает:
        str: Предобработанный текст.
    """
    stop_words = set(stopwords.words('russian'))
    wnl = WordNetLemmatizer()
    stm = SnowballStemmer('russian')

    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    words = [w for w in words if w not in stop_words]
    words = [wnl.lemmatize(w) for w in words]
    words = [stm.stem(w) for w in words]
    processed_text = ' '.join(words)

    return processed_text

def main(test_file, lr_model_file, lgbm_model_file, lr_predictions_file, lgbm_predictions_file):
    test_df = pd.read_csv(test_file, delimiter='\t')

    processed_text = test_df['title'].apply(text_preprocessing).reset_index(drop=True)

    # Load the trained models
    lr_model = joblib.load(lr_model_file)
    lgbm_model = joblib.load(lgbm_model_file)

    # Predict the labels and probabilities using the logistic regression model
    test_df['is_fake_lr'] = lr_model.predict(processed_text)
    test_df['prob_fake_lr'] = lr_model.predict_proba(processed_text)[:, 1]

    # Predict the labels and probabilities using the LightGBM model
    test_df['is_fake_lgbm'] = lgbm_model.predict(processed_text)
    test_df['prob_fake_lgbm'] = lgbm_model.predict_proba(processed_text)[:, 1]

    # Save the predictions from the logistic regression model to a file
    lr_predictions_df = test_df[['is_fake_lr', 'prob_fake_lr']]
    lr_predictions_df.to_csv(lr_predictions_file, index=False)

    # Save the predictions from the LightGBM model to a file
    lgbm_predictions_df = test_df[['is_fake_lgbm', 'prob_fake_lgbm']]
    lgbm_predictions_df.to_csv(lgbm_predictions_file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction by LinearRegressuin and LightGBM ')
    parser.add_argument('test_file', type=str, help='Path to the test file')
    parser.add_argument('lr_model_file', type=str, help='Path to the logistic regression model file')
    parser.add_argument('lgbm_model_file', type=str, help='Path to the LightGBM model file')
    parser.add_argument('lr_predictions_file', type=str, help='Path to save the logistic regression predictions')
    parser.add_argument('lgbm_predictions_file', type=str, help='Path to save the LightGBM predictions')
    args = parser.parse_args()

    main(args.test_file, args.lr_model_file, args.lgbm_model_file, args.lr_predictions_file, args.lgbm_predictions_file)
