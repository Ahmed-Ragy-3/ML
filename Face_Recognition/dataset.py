import os
import numpy as np
from PIL import Image


class Dataset:
    def __init__(self, data_path):
        self.data_path = data_path
        self.D = None  # Data Matrix
        self.y = None  # Label Vector
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None

    def load_data(self):
        data_matrix = []
        labels = []

        # The database has 40 subjects
        for subject_id in range(1, 41):
            subject_folder = os.path.join(self.data_path, f's{subject_id}')

            # There are 10 images per subject
            for i in range(1, 11):
                img_path = os.path.join(subject_folder, f'{i}.pgm')

                with Image.open(img_path) as img:
                    # Every image is grayscale of size 92x112
                    # Flatten into a vector of 10304 values
                    img_vector = np.array(img).flatten()
                    data_matrix.append(img_vector)
                    # Labels are integers from 1:40
                    labels.append(subject_id)

        # Stack into a single Data Matrix D (400x10304)
        self.D = np.array(data_matrix)
        self.y = np.array(labels)
        return self.D, self.y

    def split_data(self):
        if self.D is None:
            self.load_data()

        # Training set: Odd rows (1, 3, 5...) which are indices 0, 2, 4...
        self.X_train = self.D[0::2]
        self.y_train = self.y[0::2]

        # Test set: Even rows (2, 4, 6...) which are indices 1, 3, 5...
        self.X_test = self.D[1::2]
        self.y_test = self.y[1::2]

        # This results in 5 instances per person for training and 5 for testing
        return self.X_train, self.y_train, self.X_test, self.y_test


# Example Usage:
if __name__ == "__main__":
    path = "./Dataset"
    orl = Dataset(path)

    X_train, y_train, X_test, y_test = orl.split_data()

    print(f"Data Matrix D: {orl.D.shape}")
    print(f"Train Shape: {X_train.shape}")  # Should be (200, 10304)
    print(f"Test Shape: {X_test.shape}")  # Should be (200, 10304)