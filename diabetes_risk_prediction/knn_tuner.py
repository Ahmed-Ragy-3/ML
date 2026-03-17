import pickle
from dataset import Dataset
from knn import KNN
from sklearn.metrics import accuracy_score, f1_score

K_VALUES       = [3, 5, 11, 21]
DISTANCES      = ['euclidean_distance', 'manhattan_distance']
BALANCE_METHODS = [None, 'oversample', 'undersample']
PCA_COMPONENTS = [5, 10, 15, 21]

def knn_tuner(dataset_path, results_file="knn_results.pkl", best_file="knn_best_config.pkl"):
    results = []

    for balance_method in BALANCE_METHODS:
        for n_components in PCA_COMPONENTS:

            # Rebuild dataset with this pca_components value
            ds = Dataset(path=dataset_path, feature_scale=True, feature_selection=True, pca_components=n_components)
            ds.prepare()
            if balance_method:
                ds.handle_imbalance(balance_method)

            knn = KNN(dataset=ds)

            for k in K_VALUES:
                for dist in DISTANCES:
                    knn.set_k(k)
                    knn.set_dist(dist)

                    y_val_pred = knn.predict(ds.x_val)

                    acc         = accuracy_score(ds.y_val, y_val_pred)
                    f1_micro    = f1_score(ds.y_val, y_val_pred, average='micro')
                    f1_macro    = f1_score(ds.y_val, y_val_pred, average='macro')
                    f1_weighted = f1_score(ds.y_val, y_val_pred, average='weighted')

                    results.append({
                        'balance_method': balance_method,
                        'pca_components': n_components,
                        'k':              k,
                        'distance':       dist,
                        'accuracy':       acc,
                        'f1_micro':       f1_micro,
                        'f1_macro':       f1_macro,
                        'f1_weighted':    f1_weighted
                    })

                    print(f"[{balance_method}] pca={n_components}, k={k}, dist={dist} -> acc={acc:.4f}, f1_micro={f1_micro:.4f}, f1_macro={f1_macro:.4f}, f1_weighted={f1_weighted:.4f}")

    # Find the best config by f1_macro
    best = max(results, key=lambda x: x['f1_macro'])
    print("\nBest Configuration:")
    print(best)

    # Save full results
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)

    # Save best configuration
    with open(best_file, 'wb') as f:
        pickle.dump(best, f)

    print(f"\nResults saved to '{results_file}'")
    print(f"Best config saved to '{best_file}'")

dataset_path = r"D:\Faculty Of Engineering\Level 3\Second Term\Pattern Recognition\Labs\ML\diabetes_risk_prediction\data\diabetes_012_health_indicators_BRFSS2015.csv"
knn_tuner(dataset_path)