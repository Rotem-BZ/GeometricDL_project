from os.path import join
import pickle
import joblib
import time
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import torch
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm

TENSORS_FOLDER = '/home/niv.ko/Geometric/saved_data'
# FEATURES_PATH = join(DATA_FOLDER, 'extracted_features.pt')
# LABELS_PATH = join(DATA_FOLDER, 'optimal_depths.pt')
TRAINED_NETWORK_PATH = 'saved_metrics/lightning_logs/version_0/checkpoints/epoch=2-step=267.ckpt'
DEPTH_MODEL_PATH = 'depth_model.pkl'


def load_data(depth, combined, set_name):
    set_suffix = {'train': '_train',
                  'val': '_val',
                  'test': ''}[set_name]
    features_file = f'{depth}_layers_{"combined_" if combined else ""}features{set_suffix}.pt'
    with open(join(TENSORS_FOLDER, features_file), 'rb') as file:
        X = torch.load(file).numpy()
    depths_file = f'{depth}_layers_optimal_depths{set_suffix}.pt'
    with open(join(TENSORS_FOLDER, depths_file), 'rb') as file:
        y = torch.load(file).numpy()

    return X, y


def train_depth_classifier(depth, combined_features, set_name):
    X, y = load_data(depth, combined_features, set_name)
    # X, y = RandomUnderSampler().fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, y_train = RandomUnderSampler().fit_resample(X_train, y_train)
    candidate_models = {
        'decision_tree': lambda: tree.DecisionTreeClassifier(),
        'boosted_tree': lambda: AdaBoostClassifier(n_estimators=500),
        'random_forest': lambda: RandomForestClassifier(500),
        'MLP': lambda: MLPClassifier(hidden_layer_sizes=(50, 50, 50, 50), random_state=42, max_iter=3000)
    }
    if len(candidate_models) > 1:
        scores = {'mean': dict(), 'std': dict()}
        for name, clf_func in tqdm(candidate_models.items(), 'CV all models'):
            if name == 'MLP':
                scaler = StandardScaler()
                scaler.fit(X_train)
                new_X = scaler.transform(X_train)
                scores_array = cross_val_score(clf_func(), new_X, y_train, cv=5)
            else:
                scores_array = cross_val_score(clf_func(), X_train, y_train, cv=5)
            scores['mean'][name] = scores_array.mean()
            scores['std'][name] = scores_array.std()

        with open(join(TENSORS_FOLDER, f'depth_{depth}_model_comparison.pkl'), 'wb') as file:
            pickle.dump(scores, file)

        pprint(scores)

        best_clf_name, _ = max(scores['mean'].items(), key=lambda tup: tup[1])
    else:
        best_clf_name, = candidate_models.keys()
    if best_clf_name == 'MLP':
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    clf = candidate_models[best_clf_name]()
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    t1 = time.perf_counter()
    print(f"chosen model: {best_clf_name}.\nTime to fit: {t1 - t0}")
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    plt.hist(clf.predict(X_test))
    plt.title('preds hist')
    plt.savefig(join(TENSORS_FOLDER, 'preds_hist.png'))
    plt.clf()
    plt.hist(y_test)
    plt.title('labels hist')
    plt.savefig(join(TENSORS_FOLDER, 'labels_hist.png'))
    print(f"training score:\t{train_score}\ntest score:\t{test_score}")
    # joblib.dump(clf, DEPTH_MODEL_PATH)
    # print(f"saved trained model to path:\t{DEPTH_MODEL_PATH}")
    # joblib.load(DEPTH_MODEL_PATH)


def analyze_centrality_measures(depth, experiment_type: str):
    X, y = load_data(depth)
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
        selector = SelectKBest(f_classif, k=4)
        selector.fit(X, y)
        scores = -np.log10(selector.pvalues_)
        scores /= scores.max()

        X_indices = np.arange(X.shape[-1])
        plt.figure(1)
        plt.clf()
        plt.bar(X_indices - 0.05, scores, width=0.2)
        plt.title("Feature univariate score")
        plt.xlabel("Feature number")
        plt.ylabel(r"Univariate score ($-Log(p_{value})$)")
        plt.savefig()
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
    # analyze_centrality_measures('decision_tree')
    train_depth_classifier(7, combined_features=False, set_name='train')
