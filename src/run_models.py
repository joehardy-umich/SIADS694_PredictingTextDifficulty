RANDOM_SEED = 1337


def load_data(input_data_file, labels_file, stage, model_type=['full', 'baseline'][0]):
    X = pd.read_pickle(input_data_file)
    y = pd.read_pickle(labels_file)

    if stage == '1A':
        print("Subsetting data for stage 1A or baseline...")
        random_indices = np.random.choice(range(len(X)), 10000, replace=False)
        X = X.iloc[random_indices]
        y = y.iloc[random_indices]
    elif model_type == 'baseline':
        X[pd.isnull(X)] = 0.
        print("Subsetting data for stage 1A or baseline...")
        random_indices = np.random.choice(range(len(X)), 10000, replace=False)
        X = X.iloc[random_indices]
        y = y.iloc[random_indices]
    else:
        X[pd.isnull(X)] = 0.
    print(X.shape[1])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    return X_train, X_val, y_train, y_val


def train_models(X_train, y_train, stage, model_type=['full', 'baseline'][0]):
    if stage == '1A' or model_type == 'baseline':
        models = [LogisticRegression(random_state=RANDOM_SEED, solver='lbfgs')]

    elif stage == '1B':
        models = [LogisticRegression(random_state=RANDOM_SEED, solver='lbfgs', verbose=1, max_iter=500),
                  KNeighborsClassifier(n_neighbors=75, n_jobs=-1),
                  GaussianNB(priors=[0.5, 0.5]),
                  SGDClassifier(random_state=RANDOM_SEED, verbose=1, max_iter=500, n_iter_no_change=100, tol=1e-4,
                                loss='log'),
                  MLPClassifier(random_state=RANDOM_SEED, verbose=1, solver='adam',
                                # activation='relu', learning_rate='adaptive',
                                hidden_layer_sizes=(3, 2), max_iter=500),
                  RandomForestClassifier(random_state=RANDOM_SEED, verbose=1, max_depth=None,
                                         n_jobs=-1, n_estimators=1000,
                                         criterion='gini')
                  ]
    elif stage == '2':
        models = [LogisticRegression(random_state=RANDOM_SEED, solver='lbfgs', verbose=1, max_iter=300),
                  KNeighborsClassifier(n_neighbors=75, n_jobs=-1),
                  GaussianNB(priors=[0.5, 0.5]),
                  SGDClassifier(random_state=RANDOM_SEED, verbose=1, max_iter=500, n_iter_no_change=100, tol=1e-4,
                                loss='log'),
                  MLPClassifier(random_state=RANDOM_SEED, verbose=1, solver='adam',
                                # activation='relu', learning_rate='adaptive',
                                hidden_layer_sizes=(200, 150, 50), max_iter=200),
                  RandomForestClassifier(random_state=RANDOM_SEED, verbose=1, max_depth=None,
                                         n_jobs=-1, n_estimators=1000,
                                         criterion='gini')
                  ]
    elif stage == '3':
        models = [LogisticRegression(random_state=RANDOM_SEED, solver='lbfgs', verbose=1, max_iter=300),
                  KNeighborsClassifier(n_neighbors=75, n_jobs=-1),
                  GaussianNB(priors=[0.5, 0.5]),
                  SGDClassifier(random_state=RANDOM_SEED, verbose=1, max_iter=500, n_iter_no_change=100, tol=1e-4,
                                loss='log'),
                  MLPClassifier(random_state=RANDOM_SEED, verbose=1, solver='adam',
                                # activation='relu', learning_rate='adaptive',git
                                hidden_layer_sizes=(200, 150, 50), max_iter=200),
                  RandomForestClassifier(random_state=RANDOM_SEED, verbose=1, max_depth=None,
                                         n_jobs=-1, n_estimators=1000,
                                         criterion='gini')
                  ]
    else:
        models = []
    if models:
        for model in models:
            model.fit(X_train, y_train)

    return models


def get_metrics(models, X_val, y_val, X_train, y_train):
    metrics_dict = {}
    if models:
        for model in models:
            n = str(model)
            y_val_pred = model.predict(X_val)
            y_train_pred = model.predict(X_train)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            val_accuracy = accuracy_score(y_val, y_val_pred)

            metrics_dict[n] = {
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy
            }

    return metrics_dict


if __name__ == '__main__':
    import argparse
    import pickle
    import os
    import numpy as np
    import pandas as pd
    import json

    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import SGDClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    from sklearn.metrics import accuracy_score

    parser = argparse.ArgumentParser()

    # INPUTS
    parser.add_argument('input_data_file', help='file containing input feature representation')
    parser.add_argument('labels_file', help='file containing input labels')

    # OUTPUTS
    parser.add_argument('output_baseline_models_file', help='file to contain trained baseline models')
    parser.add_argument('output_models_file', help='file to contain trained models')

    # METRICS
    parser.add_argument('baseline_metrics_file', help='file to contain baseline metrics')
    parser.add_argument('metrics_file', help='file to contain metrics')

    args = parser.parse_args()
    stage = os.path.splitext(os.path.split(args.input_data_file)[1])[0].replace("stage", "")

    print("Stage %s: Load data..." % stage)
    baseline_X_train, baseline_X_val, baseline_y_train, baseline_y_val = load_data(args.input_data_file,
                                                                                   args.labels_file,
                                                                                   stage, model_type='baseline')
    X_train, X_val, y_train, y_val = load_data(args.input_data_file, args.labels_file, stage, model_type='full')

    print("Stage %s: Training models..." % stage)
    baseline_models = train_models(baseline_X_train, baseline_y_train, stage, model_type='baseline')
    models = train_models(X_train, y_train, stage, model_type='full')

    print("Stage %s: Getting metrics..." % stage)
    baseline_metrics = get_metrics(baseline_models, baseline_X_val, baseline_y_val, baseline_X_train, baseline_y_train)
    metrics = get_metrics(models, X_val, y_val, X_train, y_train)

    pickle.dump(baseline_models, open(args.output_baseline_models_file, 'wb'))
    pickle.dump(models, open(args.output_models_file, 'wb'))

    with open(args.baseline_metrics_file, 'w') as metrics_file:
        json.dump(baseline_metrics, metrics_file)
    with open(args.metrics_file, 'w') as metrics_file:
        json.dump(metrics, metrics_file)
