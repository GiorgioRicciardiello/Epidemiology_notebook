import warnings
import pathlib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, accuracy_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, FunctionTransformer
from skorch.callbacks import Callback, ProgressBar
from skorch.net import NeuralNet
import torch
from torch import nn
warnings.simplefilter('ignore')

if __name__ == '__main__':
    data_path = r'C:\Users\giorg\OneDrive - Fundacion Raices Italo Colombianas\projects\Epidemiology_notebook\misc\data\winequality-red.csv'
    data = pd.read_csv(data_path)

    # StandardScaler on the more gaussian-like columns and a PowerTransformer on the other ones.
    gaussian_columns = ['alcohol', 'chlorides', 'fixed acidity',
                        'density',
                        'pH', 'sulphates', 'volatile acidity']
    power_columns = ['citric acid', 'free sulfur dioxide', 'residual sugar',
                     'total sulfur dioxide']

    column_transformer = ColumnTransformer([
        ('gaussian', StandardScaler(), gaussian_columns),
        ('power', PowerTransformer(), power_columns)
    ])

    X_trans = column_transformer.fit_transform(data)

    import requests
    from contextlib import closing
    import csv
    from codecs import iterdecode

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

    import urllib.request

    file_name = "winequality-red.csv"

    urllib.request.urlretrieve(url, file_name)

    print(f"File {file_name} downloaded successfully.")