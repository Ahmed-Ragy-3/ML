import numpy as np
from sklearn.discriminant_analysis import StandardScaler

LABEL = "Diabetes_012"

class Dataset:
   RANDOM_SEED = 42
   TEST_SIZE = 0.2
   VAL_SIZE = 0.1

   def __init__(self, path):
      self.path = path
      self.data = None

   def load_data(self):
      import pandas as pd
      self.data = pd.read_csv(self.path)
      return self.data

   def split(self):
      """Split the dataset into 70% training, 10% validation, and
			20% testing. Use stratified splitting so that each subset maintains the same original
			class distribution.

      Returns:
			_type_: _description_
      """
      from sklearn.model_selection import train_test_split
      train_data, temp_data = train_test_split(
			self.data,
			test_size=self.TEST_SIZE + self.VAL_SIZE,
			stratify=self.data[LABEL],
			random_state=self.RANDOM_SEED
      )
      val_data, test_data = train_test_split(
			temp_data,
			test_size=self.VAL_SIZE,
			stratify=temp_data[LABEL],
			random_state=self.RANDOM_SEED
      )
      return train_data, val_data, test_data

   def feature_scaling(x_train, x_val, x_test):
      """
      Standardize features using Z-score normalization (StandardScaler).

      This function applies feature scaling to ensure all features have:
      - Mean (μ) = 0
      - Standard Deviation (σ) = 1

      Formula applied to each feature: X_scaled = (X - μ) / σ
      """
      from sklearn.preprocessing import StandardScaler
      scaler = StandardScaler()
      x_train_scaled = scaler.fit_transform(x_train)
      x_val_scaled = scaler.transform(x_val)
      x_test_scaled = scaler.transform(x_test)
      return x_train_scaled, x_val_scaled, x_test_scaled

   def dimensionality_reduction(self, x_train, x_val, x_test):
      """Dimensionality Reduction: Apply a dimensionality reduction technique (e.g., PCA) to
                        further reduce the feature space while retaining as much variance as possible.

      Args:
                        x_train (_type_): _description_
                        x_val (_type_): _description_
                        x_test (_type_): _description_
      """
      from sklearn.decomposition import PCA
      pca = PCA(n_components=10)
      x_train_pca = pca.fit_transform(x_train)
      x_val_pca = pca.transform(x_val)
      x_test_pca = pca.transform(x_test)
      return x_train_pca, x_val_pca, x_test_pca

   def oversample(self, x_train, y_train):
      from imblearn.over_sampling import SMOTE
      smote = SMOTE(random_state=self.RANDOM_SEED)
      x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
      return x_train_resampled, y_train_resampled

   def undersample(self, x_train, y_train):
      from imblearn.under_sampling import RandomUnderSampler
      undersampler = RandomUnderSampler(random_state=self.RANDOM_SEED)
      x_train_resampled, y_train_resampled = undersampler.fit_resample(x_train, y_train)
      return x_train_resampled, y_train_resampled

   def class_weighting(self, y_train):
      from sklearn.utils import class_weight
      class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
      return class_weights

   def handle_imbalance(self, x_train, y_train, method='oversample'):
      match method:
         case 'oversample':
            return self.oversample(x_train, y_train)
         case 'undersample':
            return self.undersample(x_train, y_train)
         case 'class_weighting':
            return self.class_weighting(y_train)
         case _:
            raise ValueError("Invalid method. Choose from 'oversample', 'undersample', or 'class_weighting'.")
