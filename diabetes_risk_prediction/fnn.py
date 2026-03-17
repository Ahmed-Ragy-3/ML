from classifier import Classifier
from dataset import Dataset

class FNN(Classifier):
	def __init__(self, dataset: Dataset, hidden_layers=[64, 32, 16], activations=['relu', 'relu', 'relu'], l2=0.0, dropout=0.0):
		self.dataset: Dataset = dataset
		self.hidden_layers = hidden_layers
		self.activations = activations if activations else []
		self.l2 = l2
		self.dropout = dropout
		self.model = self.build_model()

	def build_model(self):
		import tensorflow as tf

		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Input(shape=(self.dataset.input_dim(),)))

		for i in range(len(self.hidden_layers)):
			activation = self.activations[i] if i < len(self.activations) else 'relu'

			model.add(tf.keras.layers.Dense(
				self.hidden_layers[i],
				activation=activation,
				kernel_regularizer=tf.keras.regularizers.l2(self.l2)
			))

			# Optional upgrade
			model.add(tf.keras.layers.BatchNormalization())

			if self.dropout > 0:
				model.add(tf.keras.layers.Dropout(self.dropout))

		model.add(tf.keras.layers.Dense(3, activation='softmax'))

		model.compile(
			optimizer='adam',
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy']
		)

		return model

	def fit(self, epochs=50, batch_size=32, callbacks=None):
		import tensorflow as tf

		early_stopping = tf.keras.callbacks.EarlyStopping(
			monitor='val_loss',
			patience=5,
			restore_best_weights=True
		)

		all_callbacks = [early_stopping] + (callbacks or [])

		history = self.model.fit(
			self.dataset.x_train, self.dataset.y_train,
			validation_data=(self.dataset.x_val, self.dataset.y_val),
			epochs=epochs,
			batch_size=batch_size,
			callbacks=all_callbacks,
			verbose=1
		)
		return history

	def predict(self, X):
		import numpy as np
		probs = self.model.predict(X)
		return np.argmax(probs, axis=1)