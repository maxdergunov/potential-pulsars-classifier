import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, make_scorer, matthews_corrcoef, cohen_kappa_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from ast import literal_eval
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

import time


def choose_dataset(pca=False, tuning=False, scaling=True, sampling='smote'):
    """Chooses and returns a proper pre-made .csv file"""
    y_test = pd.read_csv('./data/y_test.csv')
    X_train_path, y_train_path, X_test_path = './data/X_train', './data/y_train', './data/X_test'
    
    if scaling == True:
        X_train_path += '_scaled'
        y_train_path += '_scaled'
        X_test_path += '_scaled'
    if pca == True:
        X_train_path += '_pca'
        y_train_path += '_pca'
        X_test_path += '_pca'
    if sampling != False:
        X_train_path += '_' + sampling
        y_train_path += '_' + sampling
    
    X_train_path += '.csv'
    y_train_path += '.csv'
    X_test_path += '.csv'
    X_train = pd.read_csv(X_train_path, index_col=False)
    y_train = pd.read_csv(y_train_path, index_col=False)
    X_test = pd.read_csv(X_test_path, index_col=False)
    
    print('Loaded', X_train_path)
    print('Loaded', y_train_path)
    print('Loaded', X_test_path)
    return X_train, y_train, X_test, y_test
    
def choose_algorithm(algo='lr'):
    """Returns algorithm and parameters grid"""
    if algo == 'lr':
        grid = {'penalty': ['l1', 'l2'],
                'C': np.linspace(0.1, 10, 50),
                'solver':['liblinear', 'newton-cg', 'lbfgs'],
                'max_iter': [50, 100, 500]}
        algorithm = LogisticRegression(random_state=25)
                    
    elif algo == 'rf':
        grid = {'n_estimators': [400, 200, 100],
                'criterion': ['entropy'],
                'max_depth':[10, 9, 8, 7],
                'max_features': ['auto', 'log2']}
        algorithm = RandomForestClassifier(random_state=25)
                    
    elif algo == 'abc':
        grid = {'n_estimators': range(500, 1100, 100),
                'algorithm': ['SAMME', 'SAMME.R'],
                'learning_rate': np.linspace(0.01, 0.3, 5)}
        algorithm = AdaBoostClassifier(random_state=25)
                    
    elif algo == 'etc':
        grid = {'n_estimators': [50, 100, 500, 1000],
                'criterion': ['entropy'],
                'max_depth':range(5, 9),
                'max_features': ['auto', 'log2', None]}
        algorithm = ExtraTreesClassifier(random_state=25)
                        
    elif algo == 'xgbc':
        grid = {'max_depth': [3, 4],
                'learning_rate': [0.3],
                'booster': ['dart'],
                'subsample': [0.7],
                 'colsample_bylevel': [0.7, 0.6], 
                'colsample_bynode': [0.7, 0.6], 
                'colsample_bytree': [0.7, 0.6],
                'gamma': [0],
                'n_estimators': [1000, 100, 50],
                'tree_method': ['exact'],
                'rate_drop': [0.03]}
        algorithm = XGBClassifier(random_state=25)
                        
    elif algo == 'svc':
        grid = {'C':np.geomspace(0.8, 1.2, 5),
                'kernel': ['linear', 'poly', 'rbf'],
                'gamma': ['scale', 'auto']}
        algorithm = SVC(random_state=25)
        
    elif algo == 'dt':
        grid = {'splitter':['best', 'random'],
                'max_depth':range(3, 9),
                'max_features': ['auto', 'log2', None]}
        algorithm = DecisionTreeClassifier(random_state=25)
        
    elif algo == 'knn':
        grid = {'n_neighbors': range(3, 15),
               'weights': ['uniform', 'distance'],
               'algorithm': ['ball_tree', 'kd_tree', 'brute']}       
        algorithm = KNeighborsClassifier()
    print('Algorithm selected.')
    return algorithm, grid


def tuner(algorithm, grid, X_train, y_train, scoring='cohen', n_splits=5, verbose=0, n_jobs=-1):
    """Performs a gridsearch cross-validation and returns the best estimator"""
    print('Tuning begins...')
    if scoring == 'cohen':
        scorer = make_scorer(cohen_kappa_score)
    elif scoring == 'matthews':
        scorer = make_scorer(matthews_corrcoef)
    elif scoring == 'f1':
        scorer = make_scorer(f1_score)
    elif scoring == 'recall':
        scorer = make_scorer(recall_score)
    elif scoring == 'precision':
        scorer = make_scorer(precision_score)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    gridsearch = GridSearchCV(algorithm, grid, cv=skf, n_jobs=n_jobs, scoring=scorer, verbose=verbose)
    gridsearch.fit(X_train, np.ravel(y_train))
    algorithm = gridsearch.best_estimator_
    
    print('Tuning completed.')
   
    return algorithm




def train_model(algo='lr', pca=False, tuning=False, scaling=True, sampling='smote', n_jobs=-1, scoring='cohen', n_splits=5, verbose=0):
    """Loads .csv file based on choosed parameters, chooses algorithm, performs tuning if necessary, 
    fits algorithm and returns estimator and predictions"""
    print('Training begins...')
    X_train, y_train, X_test, y_test = choose_dataset(pca, tuning, scaling, sampling)
    alg, grid = choose_algorithm(algo=algo)
    if tuning==True:
        algorithm = tuner(alg, grid, X_train, y_train, scoring=scoring, n_splits=n_splits, verbose=verbose, n_jobs=n_jobs)
    else:
        algorithm = alg
    
    algorithm.fit(X_train, y_train)
    print('Training completed.')
    preds = algorithm.predict(X_test)
    return algorithm, preds, y_test #надо убрать y_test, а то ж он один
    
    
    
def add_new_row(algorithm, preds, y_test):
    """Collects various parameters for the given estimator and returns it as dictionary"""
    report = classification_report(y_test, preds, output_dict=True)
    
    non_pulsar_precision = report['0']['precision']
    non_pulsar_recall = report['0']['recall']
    non_pulsar_f1 = report['0']['f1-score']
    pulsar_precision = report['1']['precision']
    pulsar_recall = report['1']['recall']
    pulsar_f1 = report['1']['f1-score']
    weighted_precision = report['weighted avg']['precision']
    weighted_recall = report['weighted avg']['recall']
    weighted_f1 = report['weighted avg']['f1-score']
    matthews_coef = round(matthews_corrcoef(y_test, preds), 3)
    cohens_kappa = round(cohen_kappa_score(y_test, preds), 3)
    parameters = algorithm.get_params()
    
    new_row = {'Non-pulsar precision': non_pulsar_precision,
              'Non-pulsar recall': non_pulsar_recall,
              'Non-pulsar F1': non_pulsar_f1,
              'Pulsar precision': pulsar_precision,
              'Pulsar recall': pulsar_recall,
              'Pulsar F1': pulsar_f1,
              'Weighted precision': weighted_precision,
              'Weighted recall': weighted_recall,
              'Weighted F1': weighted_f1,
              'Matthews corrcoefficient': matthews_coef,
              "Cohen's Kappa": cohens_kappa,
              'Parameters': parameters}
    
    return new_row

def plot_conf_matrix(parameters, pca, scaling, sampling, algorithm, place, save=False): 
    y_test = pd.read_csv('./data/y_test.csv')
    X_test_path = './data/X_test'
    
    X_train_path, y_train_path, X_test_path = './data/X_train', './data/y_train', './data/X_test'
    
    if scaling == True:
        X_train_path += '_scaled'
        y_train_path += '_scaled'
        X_test_path += '_scaled'
    if pca == True:
        X_train_path += '_pca'
        y_train_path += '_pca'
        X_test_path += '_pca'
    if sampling != False:
        X_train_path += '_' + sampling
        y_train_path += '_' + sampling
    
    X_train_path += '.csv'
    y_train_path += '.csv'
    X_test_path += '.csv'
    
    X_train = pd.read_csv(X_train_path, index_col=False)
    y_train = pd.read_csv(y_train_path, index_col=False)
    X_test = pd.read_csv(X_test_path, index_col=False)
                
    if algorithm == 'Logistic Regression':
            algo = LogisticRegression(**literal_eval(parameters))
    elif algorithm == 'Random Forest':
            algo = RandomForestClassifier(**literal_eval(parameters))
    elif algorithm == 'AdaBoost':
            algo = AdaBoostClassifier(**literal_eval(parameters))
    elif algorithm == 'ExtraTreeClassifier':
            algo = ExtraTreesClassifier(**literal_eval(parameters))
    elif algorithm == 'Gradient Boosting':
            parameters = parameters.replace("'missing': nan,", '')
            algo = XGBClassifier(**literal_eval(parameters))
    elif algorithm == 'SVC':
            algo = SVC(**literal_eval(parameters))
    elif algorithm == 'Decision Tree':
            algo = DecisionTreeClassifier(**literal_eval(parameters))
    elif algorithm == 'KNN':
            algo = KNeighborsClassifier(**literal_eval(parameters))
                
    algo.fit(X_train, y_train)
        
    if place == 1:
        title = '1st place - {}'.format(algorithm)
    elif place == 2:
        title = '2nd place - {}'.format(algorithm)
    elif place == 3:
        title = '3rd place - {}'.format(algorithm)
    else:
        title = '{}th place - {}'.format(place, algorithm)
        
    plot_confusion_matrix(algo, X_test, y_test, values_format='g')
    plt.title(title)
    
    if save == True:
        plt.savefig('./images/{}.png'.format(title), dpi=150, format='png')
        
    plt.show() 