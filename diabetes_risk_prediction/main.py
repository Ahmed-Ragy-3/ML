
from dataset import LABEL, Dataset
from fnn import FNN
from softmax import Softmax
# from knn import KNN
from evaluation import ModelEvaluator

dataset_path = "C:\\COLLEGE\\Term_8\\pattern\\ML\\diabetes_risk_prediction\\data\\diabetes_012_health_indicators_BRFSS2015.csv"  # Update with the actual path to your dataset

def main():
	print("Testing KNN Classifier on Diabetes Dataset...")
	# dataset_path = "./data/diabetes_012_health_indicators_BRFSS2015.csv"
	dataset = Dataset(dataset_path, imbalance_method='oversample', feature_scale=True, feature_selection=True)
	dataset.prepare()

	# Choose distance metric: euclidean_distance or manhattan_distance
	# knn = KNN(dist='euclidean_distance', k=5)
	# knn.fit(x_train_scaled, y_train)

	# softmax = Softmax(input_dim=x_train_scaled.shape[1], l2=0.0)
	# softmax.fit(x_train_scaled, y_train)

	fnn = FNN(input_dim=x_train_scaled.shape[1], hidden_layers=[64, 32, 16], activations=['relu', 'relu', 'relu'], l2=0.01, dropout=0.2)
	fnn.fit(x_train_scaled, y_train, x_val_scaled, y_val, epochs=20, batch_size=32)

	evaluator = ModelEvaluator(fnn, x_test_scaled, y_test)
	results = evaluator.evaluate()
	print("Evaluation Results:")
	for metric, value in results.items():
		print(f"{metric}: {value:.4f}")
	evaluator.plot_confusion_matrix()

if __name__ == "__main__":
	main()