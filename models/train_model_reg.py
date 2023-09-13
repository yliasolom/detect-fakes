import os
import click
import pandas as pd
import numpy as np
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
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from wordcloud import WordCloud
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
stop_words = set(stopwords.words('russian'))

RANDOM_STATE = 1052023
import warnings
warnings.filterwarnings('ignore')

from utils import hm

@click.command()
@click.option('--train-file', required=True, help='Path to the train file')
def train_models(train_file):

    df = pd.read_csv(train_file, delimiter='\t')
    print('Duplicates count:', df['title'].duplicated().sum())
    df.isnull().sum()
    df.sample(5)

    def text_preprocessing(text):
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

    corpus = df['title'].apply(text_preprocessing).reset_index(drop=True)
    corpus

    plt.subplots(figsize=(12, 12))
    wordcloud = WordCloud(background_color="white", width=1024, height=768).generate(" ".join(corpus))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig("reports/wordcloud.png")

    X_train, X_test, y_train, y_test = train_test_split(corpus, df['is_fake'], test_size=0.25, random_state=RANDOM_STATE)
    del corpus

    print("Shape of X_train:", X_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)

    for r in [(1, 1), (2, 2), (1, 2)]:
        print('n-gram range {} : features count = {}'.format(r, TfidfVectorizer(ngram_range=r).fit_transform(X_train).shape[1]))
        cv_params = {
            'scoring': 'f1',
            'return_train_score': True,
            'cv': 3
        }

    pipe = Pipeline([('vct', TfidfVectorizer(min_df=1e-5)),
                     ('clf', LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE, max_iter=500))
                     ])
    params = {'clf__solver': ['lbfgs', 'liblinear'],
              'clf__C': [0.1, 1, 10]
              }
    grid_lr = GridSearchCV(pipe, params, **cv_params, n_jobs=-1)
    grid_lr.fit(X_train, y_train)
    hm.best_cv_models(grid_lr, 10)

    pipe = Pipeline([('vct', TfidfVectorizer(min_df=1e-5)),
                     ('clf', LGBMClassifier(random_state=RANDOM_STATE))])

    params = {
        'clf__max_depth': np.arange(40, 70, 10),
        'clf__learning_rate': [0.1, 0.2],
        'clf__num_leaves': np.arange(40, 70, 10),
        'clf__n_estimators': [100, 200, 250]
    }

    grid_lgbm = GridSearchCV(pipe, params, **cv_params)
    grid_lgbm.fit(X_train, y_train)
    hm.best_cv_models(grid_lgbm, 10)

    best_lr = grid_lr.best_estimator_
    lgbm_unigram = grid_lgbm.best_estimator_
    lgbm_bigram = Pipeline([('vct', TfidfVectorizer(min_df=1e-5, ngram_range=(1, 2))),
                            ('clf', LGBMClassifier(random_state=RANDOM_STATE,
                                                   max_depth=50, learning_rate=0.2,
                                                   num_leaves=60, n_estimators=100))
                            ])
    models = {'LogisticRegression': best_lr,
              'LightGBM unigram': lgbm_unigram, 'LightGBM bigram': lgbm_bigram}
    scores_table = {}
    for name, model in models.items():
        scores_table[name] = hm.test_model(
            model, X_train, X_test, y_train, y_test, f1_score)

    hm.roc_curve_plot(models, X_test, y_test)
    pd.DataFrame(scores_table)

    # Create the models directory if it doesn't exist
    models_dir = 'weights/basic'
    os.makedirs(models_dir, exist_ok=True)

    # Save the models
    lr_model_file = os.path.join(models_dir, 'logistic_regression.pkl')
    lgbm_model_file = os.path.join(models_dir, 'lightgbm.pkl')

    joblib.dump(best_lr, lr_model_file)
    joblib.dump(lgbm_unigram, lgbm_model_file)

    print('Models saved successfully!')


if __name__ == '__main__':
    train_models()
