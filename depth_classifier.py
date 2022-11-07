from os.path import join
import pickle
import joblib
import time

import torch
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

DATA_FOLDER = 'saved_data'
FEATURES_PATH = join(DATA_FOLDER, 'extracted_features.pt')
LABELS_PATH = join(DATA_FOLDER, 'optimal_depths.pt')
TRAINED_NETWORK_PATH = 'saved_metrics/lightning_logs/version_0/checkpoints/epoch=2-step=267.ckpt'
DEPTH_MODEL_PATH = 'depth_model.pkl'


def load_data():
    with open(FEATURES_PATH, 'rb') as file:
        X = torch.load(file).numpy()
    with open(LABELS_PATH, 'rb') as file:
        y = torch.load(file).numpy()

    return X, y


def train_depth_classifier():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    candidate_models = {'decision_tree': tree.DecisionTreeClassifier,
                        'boosted_tree': AdaBoostClassifier,
                        'random_forest': RandomForestClassifier,
                        'MLP': lambda: MLPClassifier(hidden_layer_sizes=(20, 100, 100), random_state=42)
                        }

    scores = {'mean': dict(), 'std': dict()}
    for name, clf_func in candidate_models.items():
        if name == 'MLP':
            scalar = StandardScaler()
            scalar.fit(X_train)
            new_X = scalar.transform(X_train)
            scores_array = cross_val_score(clf_func(), new_X, y_train, cv=5)
        else:
            scores_array = cross_val_score(clf_func(), X_train, y_train, cv=5)
        scores['mean'][name] = scores_array.mean()
        scores['std'][name] = scores_array.std()

    with open(join(DATA_FOLDER, 'depth_model_comparison.pkl'), 'wb') as file:
        pickle.dump(scores, file)

    pprint(scores)

    best_clf_name, _ = max(scores['mean'].items(), key=lambda tup: tup[1])
    clf = candidate_models[best_clf_name]()
    t0 = time.perf_counter()
    clf.fit(X, y)
    t1 = time.perf_counter()
    print(f"chosen model: {best_clf_name}.\nTime to fit: {t1 - t0}")
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f"training score:\t{train_score}\ntest score:\t{test_score}")
    joblib.dump(clf, DEPTH_MODEL_PATH)
    print(f"saved trained model to path:\t{DEPTH_MODEL_PATH}")
    # joblib.load(DEPTH_MODEL_PATH)


def analyze_centrality_measures(experiment_type: str):
    X,y = load_data()
    if experiment_type == 'decision_tree':
        clf = tree.DecisionTreeClassifier()
        clf.fit(X, y)
        tree.plot_tree(clf)
        plt.show()
    elif experiment_type == 'MLP_sharpe':
        """
        use sharpe metric to evaluate contribution features to a trained NN
        """
        pass
    elif experiment_type == 'univariate selection':
        """
        use the chi-squared statistical test metric to plot a score bar graph
        """
        pass
    elif experiment_type == 'feature importance':
        """
        use the in-built feature importance function of trees in sklearn
        """
        pass
    elif experiment_type == 'correlation heatmap':
        """
        plot a heatmap with correlations between variables and the label
        """
        pass
    else:
        raise ValueError(f"illegal clf_type {experiment_type}")


if __name__ == '__main__':
    analyze_centrality_measures('decision_tree')