import pickle
from dataset import Dataset
from knn import KNN
from sklearn.metrics import accuracy_score, f1_score

K_VALUES = [3, 5, 11, 21]
DISTANCES = ['euclidean_distance', 'manhattan_distance']
BALANCE_METHODS = [None, 'oversample', 'undersample']

def knn_tuner(dataset_path, results_file="knn_results.pkl", best_file="knn_best_config.pkl"):
    results = []

    # Load and prepare dataset
    ds = Dataset(path=dataset_path, feature_scale=True)
    ds.prepare()

    for balance_method in BALANCE_METHODS:
        # Reset dataset to original before applying imbalance method
        ds.prepare()
        if balance_method:
            ds.handle_imbalance(balance_method)
        
        knn = KNN(dataset=ds)

        for k in K_VALUES:
            for dist in DISTANCES:
                # Train KNN
                knn.set_k(k)
                knn.set_dist(dist)

                # Predict on validation set
                y_val_pred = knn.predict(ds.x_val)

                # Evaluate metrics
                acc = accuracy_score(ds.y_val, y_val_pred)
                f1_micro = f1_score(ds.y_val, y_val_pred, average='micro')
                f1_macro = f1_score(ds.y_val, y_val_pred, average='macro')
                f1_weighted = f1_score(ds.y_val, y_val_pred, average='weighted')

                # Save results
                results.append({
                    'balance_method': balance_method,
                    'k': k,
                    'distance': dist,
                    'accuracy': acc,
                    'f1_micro': f1_micro,
                    'f1_macro': f1_macro,
                    'f1_weighted': f1_weighted
                })

                print(f"[{balance_method}] k={k}, dist={dist} -> acc={acc:.4f}, f1_micro={f1_micro:.4f}, f1_macro={f1_macro:.4f}, f1_weighted={f1_weighted:.4f}")

    # Find the best config by accuracy
    best = max(results, key=lambda x: x['accuracy'])
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

dataset_path = "C:\\COLLEGE\\Term_8\\pattern\\ML\\diabetes_risk_prediction\\data\\diabetes_012_health_indicators_BRFSS2015.csv"
knn_tuner(dataset_path)