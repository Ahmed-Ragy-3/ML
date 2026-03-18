import numpy as np
from sklearn.discriminant_analysis import StandardScaler

LABEL = "Diabetes_012"

class Dataset:
   RANDOM_SEED = 42
   TEST_SIZE = 0.2
   VAL_SIZE = 0.1

   def __init__(self, path, imbalance_method: str = None, feature_scale: bool = False, feature_selection: bool = False,
                pca_components: int = 10):
      self.path = path
      self.data = None

      self.imbalance_method: str = imbalance_method
      self.feature_scale: bool = feature_scale
      self.feature_selection: bool = feature_selection
      self.pca_components: int = pca_components

   def prepare(self):
      self._load_data()

      train_data, val_data, test_data = self._split()

      self.x_train = train_data.drop(columns=[LABEL]).values
      self.y_train = train_data[LABEL].values
      
      self.x_val = val_data.drop(columns=[LABEL]).values
      self.y_val = val_data[LABEL].values
      
      self.x_test = test_data.drop(columns=[LABEL]).values
      self.y_test = test_data[LABEL].values

      if self.feature_scale:
         self._feature_scaling()

      if self.feature_selection:
         self._dimensionality_reduction()

      if self.imbalance_method:
         self.handle_imbalance(method=self.imbalance_method)

   def _load_data(self):
      import pandas as pd
      self.data = pd.read_csv(self.path)

   def input_dim(self):
      return self.x_train.shape[1]

   def _split(self):
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

      val_ratio = self.VAL_SIZE / (self.TEST_SIZE + self.VAL_SIZE)

      val_data, test_data = train_test_split(
			temp_data,
			test_size=(1 - val_ratio),
			stratify=temp_data[LABEL],
			random_state=self.RANDOM_SEED
      )

      return train_data, val_data, test_data

   def _feature_scaling(self):
      """
      Standardize features using Z-score normalization (StandardScaler).

      This function applies feature scaling to ensure all features have:
      - Mean (μ) = 0
      - Standard Deviation (σ) = 1

      Formula applied to each feature: X_scaled = (X - μ) / σ
      """
      from sklearn.preprocessing import StandardScaler
      scaler = StandardScaler()
      self.x_train = scaler.fit_transform(self.x_train)
      self.x_val = scaler.transform(self.x_val)
      self.x_test = scaler.transform(self.x_test)

   def _dimensionality_reduction(self):
      """Dimensionality Reduction: Apply a dimensionality reduction technique (e.g., PCA) to
         further reduce the feature space while retaining as much variance as possible.

      Args:
         x_train (_type_): _description_
         x_val (_type_): _description_
         x_test (_type_): _description_
      """
      from sklearn.decomposition import PCA
      pca = PCA(n_components=self.pca_components)
      self.x_train = pca.fit_transform(self.x_train)
      self.x_val = pca.transform(self.x_val)
      self.x_test = pca.transform(self.x_test)

   def _oversample(self):
      from imblearn.over_sampling import SMOTE
      smote = SMOTE(random_state=self.RANDOM_SEED)
      self.x_train, self.y_train = smote.fit_resample(self.x_train, self.y_train)

   def _undersample(self):
      from imblearn.under_sampling import RandomUnderSampler
      undersampler = RandomUnderSampler(random_state=self.RANDOM_SEED)
      self.x_train, self.y_train = undersampler.fit_resample(self.x_train, self.y_train)

   def handle_imbalance(self, method='oversample'):
      match method:
         case 'oversample':
            self._oversample()
         case 'undersample':
            self._undersample()
         case _:
            raise ValueError("Invalid method. Choose from 'oversample', 'undersample', or 'class_weighting'.")
