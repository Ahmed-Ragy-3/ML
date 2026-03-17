import itertools
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from softmax import Softmax
from dataset import Dataset

def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def softmax_full_tuner_with_metrics(dataset_path,
                                    l2_values=[0.0, 0.001, 0.01, 0.1],
                                    imbalance_methods=['oversample', 'undersample', None],
                                    epochs_list=[3, 5],
                                    batch_sizes=[32, 64]):
    results = []

    # Load and prepare dataset
    ds = Dataset(path=dataset_path, feature_scale=True)
    ds.prepare()

    for imbalance, l2, epochs, batch_size in itertools.product(
        imbalance_methods, l2_values, epochs_list, batch_sizes
    ):
        ds.prepare()  # reset dataset

        if imbalance:
            ds.handle_imbalance(imbalance)

        # Train Softmax
        model = Softmax(input_dim=ds.input_dim(), l2=l2)
        model.fit(ds.x_train, ds.y_train, epochs=epochs, batch_size=batch_size)

        # Predict on validation set
        y_val_pred = model.predict(ds.x_val)

        # Metrics
        acc = accuracy_score(ds.y_val, y_val_pred)
        f1_micro = f1_score(ds.y_val, y_val_pred, average='micro')
        f1_macro = f1_score(ds.y_val, y_val_pred, average='macro')
        f1_weighted = f1_score(ds.y_val, y_val_pred, average='weighted')

        # Store results
        results.append({
            'imbalance': imbalance,
            'l2': l2,
            'epochs': epochs,
            'batch_size': batch_size,
            'accuracy': acc,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        })

        # Print all F1 scores for this combination
        print(f"[{imbalance}] L2={l2}, epochs={epochs}, batch_size={batch_size} -> "
              f"acc={acc:.4f}, f1_micro={f1_micro:.4f}, f1_macro={f1_macro:.4f}, f1_weighted={f1_weighted:.4f}")

    # Best config by accuracy
    best = max(results, key=lambda x: x['accuracy'])
    print("\nBest Hyperparameter Configuration:")
    print(best)

    # Plot confusion matrix for best config
    print("\nConfusion Matrix for Best Configuration on Validation Set:")
    ds.prepare()  # reset dataset
    if best['imbalance']:
        ds.handle_imbalance(best['imbalance'])
    best_model = Softmax(input_dim=ds.input_dim(), l2=best['l2'])
    best_model.fit(ds.x_train, ds.y_train, epochs=best['epochs'], batch_size=best['batch_size'])
    y_best_pred = best_model.predict(ds.x_val)
    plot_confusion(ds.y_val, y_best_pred, title="Confusion Matrix (Validation Set)")

    return results, best

dataset_path = "C:\\COLLEGE\\Term_8\\pattern\\ML\\diabetes_risk_prediction\\data\\diabetes_012_health_indicators_BRFSS2015.csv"
results, best_config = softmax_full_tuner_with_metrics(
    dataset_path,
    l2_values=[0.0, 0.001, 0.01, 0.1],
    imbalance_methods=['oversample', 'undersample', None],
    epochs_list=[3, 5],
    batch_sizes=[32, 64]
)