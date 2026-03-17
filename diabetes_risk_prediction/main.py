# predict_point.py
import os
import pickle
import numpy as np
from dataset import Dataset
from knn import KNN
from softmax import Softmax
from fnn import FNN

# Paths for saved models/configs
FNN_MODEL_DIR = "best_fnn_model"
SOFTMAX_MODEL_DIR = "best_softmax_model"
KNN_BEST_CONFIG_FILE = "knn_best_config.pkl"

DATASET_PATH = "C:\\COLLEGE\\Term_8\\pattern\\ML\\diabetes_risk_prediction\\data\\diabetes_012_health_indicators_BRFSS2015.csv"

def load_dataset():
    ds = Dataset(path=DATASET_PATH, feature_scale=True)
    ds.prepare()
    return ds

def load_fnn(ds: Dataset):
    import tensorflow as tf
    if os.path.exists(FNN_MODEL_DIR):
        fnn = FNN(dataset=ds)
        fnn.model = tf.keras.models.load_model(FNN_MODEL_DIR)
        return fnn
    else:
        print("No saved FNN model found!")
        return None

def load_softmax(ds: Dataset):
    import tensorflow as tf
    if os.path.exists(SOFTMAX_MODEL_DIR):
        sm = Softmax(input_dim=ds.input_dim())
        sm.model = tf.keras.models.load_model(SOFTMAX_MODEL_DIR)
        return sm
    else:
        print("No saved Softmax model found!")
        return None

def load_knn(ds: Dataset):
    if os.path.exists(KNN_BEST_CONFIG_FILE):
        with open(KNN_BEST_CONFIG_FILE, 'rb') as f:
            best_config = pickle.load(f)
        knn = KNN(dataset=ds)
        knn.set_k(best_config['k'])
        knn.set_dist(best_config['distance'])
        if best_config['balance_method']:
            ds.handle_imbalance(best_config['balance_method'])
        knn.fit(ds.x_train, ds.y_train)
        return knn
    else:
        print("No saved KNN config found!")
        return None

def predict_point(x_point: np.ndarray):
    """
    x_point: 1D numpy array of feature values (shape: [n_features])
    """
    ds = load_dataset()
    
    x_point = x_point.reshape(1, -1)  # reshape for single sample

    fnn_model = load_fnn(ds)
    softmax_model = load_softmax(ds)
    knn_model = load_knn(ds)

    predictions = {}

    if fnn_model:
        predictions['FNN'] = fnn_model.predict(x_point)[0]

    if softmax_model:
        predictions['Softmax'] = softmax_model.predict(x_point)[0]

    if knn_model:
        predictions['KNN'] = knn_model.predict(x_point)[0]

    return predictions

if __name__ == "__main__":
    # Example: a dummy feature vector
    sample_point = np.array([0.5, 0.2, 0.3, 0.1, 0.7, 0.4, 0.9, 0.8, 0.2, 0.1])
    preds = predict_point(sample_point)
    print("Predictions for input point:")
    for model_name, pred in preds.items():
        print(f"{model_name}: {pred}")