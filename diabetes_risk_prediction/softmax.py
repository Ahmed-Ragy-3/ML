import numpy as np
from diabetes_risk_prediction.classifier import Classifier


class Softmax(Classifier):
   def __init__(self, input_dim, l2=0.0):
      self.input_dim = input_dim
      self.l2 = l2
      self.model = self.build_model()

   def build_model(self):
      import tensorflow as tf

      model = tf.keras.Sequential([
			tf.keras.layers.Input(shape=(self.input_dim,)),
			tf.keras.layers.Dense(
				3,
				activation='softmax',
				kernel_regularizer=tf.keras.regularizers.l2(self.l2)
			)
      ])

      model.compile(
			optimizer='adam',
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy']
      )

      return model

   def fit(self, X, y, epochs=100, batch_size=32):
      self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

   def predict(self, X):
      probabilities = self.model.predict(X)
      return np.argmax(probabilities, axis=1)
