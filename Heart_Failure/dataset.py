import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

LABEL = "HeartDisease"

class Dataset:
   RANDOM_SEED = 42
   TEST_SIZE = 0.2
   VAL_SIZE = 0.1

   def __init__(self, path):
      self.path = path
      self.data = None

   def __str__(self):
      return f"First 50 samples:\n{self.data.head(50)}"
   
   def prepare(self):
      self._load_data()

      train_data, val_data, test_data = self._split()

      # One-hot encoding (fit on train, align others)
      train_data = pd.get_dummies(train_data)
      val_data = pd.get_dummies(val_data)
      test_data = pd.get_dummies(test_data)

      # Align columns to avoid mismatch
      val_data = val_data.reindex(columns=train_data.columns, fill_value=0)
      test_data = test_data.reindex(columns=train_data.columns, fill_value=0)

      # Split features and labels
      self.x_train = train_data.drop(columns=[LABEL])
      self.y_train = train_data[LABEL]

      self.x_val = val_data.drop(columns=[LABEL])
      self.y_val = val_data[LABEL]

      self.x_test = test_data.drop(columns=[LABEL])
      self.y_test = test_data[LABEL]

   def _load_data(self):
      self.data = pd.read_csv(self.path)

      # Optional sanity check
      if self.data.isnull().sum().sum() > 0:
         raise ValueError("Dataset contains missing values. Handle them before proceeding.")

   def _split(self):
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
