

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
		train_data, temp_data = train_test_split(self.data, test_size=self.TEST_SIZE, stratify=self.data['Outcome'], random_state=self.RANDOM_SEED)
		val_data, test_data = train_test_split(temp_data, test_size=self.VAL_SIZE, stratify=temp_data['Outcome'], random_state=self.RANDOM_SEED)
		return train_data, val_data, test_data
	
	def feature_selection(self, data):
		"""Feature Selection: The dataset contains 21 features. Implement a feature selection
			technique to identify and select the most predictive features, reducing dimensionality.

		Args:
			 data (_type_): _description_
		"""
		from sklearn.feature_selection import SelectKBest, f_classif
		X = data.drop('Outcome', axis=1)
		y = data['Outcome']
		selector = SelectKBest(score_func=f_classif, k=10)
		X_new = selector.fit_transform(X, y)
		selected_features = X.columns[selector.get_support()]
		return selected_features
	

