import os
import pandas as pd
import numpy as np
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

reports_dir = 'reports'

class hm:
    @staticmethod
    def plot_value_counts(series, n_values=25, fillna='NONE', figwidth=10):
        ''' 
        Визуализация количества встречающихся значений в pd.Series

        Параметры
        ---
        `series` : pd.Series - датафрейм
        `column` : str - название столбца
        `n_values` : int - максимальное количество значений для отображения на диаграмме
        `fillna` : Any - значение, которым необходимо заполнить пропуски
        '''
        val_counts = series.fillna(fillna).value_counts()
        bar_values = val_counts.values[:n_values]
        bar_labels = val_counts.index[:n_values].astype('str')
        plt.figure(figsize=(figwidth, 0.5*min(len(val_counts), n_values)))
        ax = sns.barplot(x=bar_values, y=bar_labels)
        ax.set(title='"{}" value counts ({} / {})'
            .format(series.name, len(bar_labels), val_counts.shape[0]),
            xlim=[0, 1.075*bar_values.max()]
            )
        plt.bar_label(ax.containers[0])
        for i in range(len(bar_labels)):
            if bar_labels[i] == fillna:
                ax.patches[i].set_color('black')
        # Save the plot as an image file
        plot_file = os.path.join(reports_dir, 'value_counts.png')
        plt.savefig(plot_file)
        plt.show()

    @staticmethod
    def best_cv_models(grid, count):
        ''' 
        Выводит таблицу с показателями моделей, показавших наилучшие значения метрики на кроссвалидации.

        Принимает  
            : `grid` - результат GridSearchCV после fit(), 
            : `count` - количество лучших моделей для вывода
        Возвращает : pd.DataFrame c параметрами моделей
        '''

        print('Estimator: {}'.format(grid.estimator))
        print('Tested {} models. Splits: {}'.format(
            len(grid.cv_results_['params']), grid.cv))
        print('Best score = {}\n'.format(grid.best_score_))
        best_idx = grid.cv_results_['rank_test_score'].argsort()[:count]

        results = {}
        results['test score'] = grid.cv_results_['mean_test_score'][best_idx]
        if 'mean_train_score' in grid.cv_results_.keys():
            results['train score'] = grid.cv_results_['mean_train_score'][best_idx]
        results['fit time, s'] = grid.cv_results_['mean_fit_time'][best_idx]
        results['score time, s'] = grid.cv_results_['mean_score_time'][best_idx]

        results_df = pd.DataFrame(results).join(
            pd.DataFrame([grid.cv_results_['params'][i] for i in best_idx])
        )
        # Save the results as a CSV file
        results_file = os.path.join(reports_dir, 'cv_results.csv')
        results_df.to_csv(results_file, index=False)

        return results_df

        

    @staticmethod
    def roc_curve_plot(models, X_test, y_test, title='ROC Curve', labels=None, figsize=(10,8)):
        ''' 
        Функция построения ROC кривой для моделей из переданнойго словаря

        Принимает:
        '''

        plt.figure(figsize=figsize)
        roc_auc_scores = {}

        for name, model in models.items():
            pred = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, pred)
            roc_auc = auc(fpr, tpr)
            roc_auc_scores[name] = roc_auc 
            plt.title(title)
            plt.plot(fpr, tpr, label='AUC = %0.4f (%s)' % (roc_auc, name))
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.grid()
        plt.xlim([0, 1]), plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        # Save the ROC curve plot as an image file
        roc_curve_file = os.path.join(reports_dir, 'roc_curve.png')
        plt.savefig(roc_curve_file)
        plt.show()

    @staticmethod
    def test_model(model, X_train, X_test, y_train, y_test, score_func=None):
        ''' 
        - Обучение модели `model` на выборках `X_train`, `y_train`
        - Предсказание обученной модели на наборе `X_test`
        - Вычисление метрики `score_func` на полученных предсказаниях и наборе `y_test`
        
        `score_func` : sklearn.metrics  

        Возвращает score, время обучения, время пердсказаний и сами предсказания в виде словаря
        '''
        # обучение
        t_beg = time.time()
        model.fit(X_train, y_train)
        time_fit = time.time() - t_beg
        # предсказания
        t_beg = time.time()
        y_pred = model.predict(X_test)
        time_predict = time.time() - t_beg
        # метрика
        score = score_func(y_test, y_pred)
        results_dict = {'score test': score,
                'fit time': time_fit,
                'predict time': time_predict
                }  
        results_df = pd.DataFrame(results_dict, index=[0])
        # Save the results as a CSV file
        results_file = os.path.join(reports_dir, 'model_results.csv')
        results_df.to_csv(results_file, index=False)

        return results_dict
