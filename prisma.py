import numpy as np
from Dataset import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class Prism:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.rules = None
    
    def fit(self, X, y):
        # Compute class probabilities
        num_samples, num_features = X.shape
        self.classes = np.unique(y)
        num_classes = len(self.classes)
        self.class_probs = np.zeros(num_classes)
        for i, cls in enumerate(self.classes):
            self.class_probs[i] = np.sum(y == cls) / num_samples
        
        # Compute feature probabilities for each class
        self.feature_probs = np.zeros((num_classes, num_features))
        for i, cls in enumerate(self.classes):
            class_samples = X[y == cls]
            total_samples = class_samples.shape[0]
            for j in range(num_features):
                feature_values, counts = np.unique(class_samples[:, j], return_counts=True)
                for value, count in zip(feature_values, counts):
                    self.feature_probs[i, j] += (count + self.alpha) / (total_samples + self.alpha * len(feature_values))
        
        # Create rules based on feature probabilities
        self.rules = []
        for i in range(num_classes):
            class_rule = []
            for j in range(num_features):
                feature_values, counts = np.unique(X[:, j], return_counts=True)
                sorted_indices = np.argsort(-self.feature_probs[i, j])
                for idx in sorted_indices:
                    if feature_values[idx] not in class_rule:
                        class_rule.append(feature_values[idx])
                        break
            self.rules.append(class_rule)
    
    def predict(self, X):
        predictions = []
        for sample in X:
            max_prob = -1
            predicted_class = None
            for i, cls in enumerate(self.classes):
                class_probs = self.class_probs[i]
                feature_probs = self.feature_probs[i]
                sample_float = np.array(sample).astype(float)
                sample_float = np.where(np.isnan(sample_float), 0, sample_float)  # Replace NaN with 0
                sample_float = np.where(sample_float == '', 0, sample_float)  # Replace empty strings with 0
                sample_float = np.clip(sample_float, 0, np.max(sample_float))  # Ensure non-negative values
                prob = np.prod(feature_probs.reshape((1, -1))[:, sample_float.astype(int)]) * class_probs
                if prob > max_prob:
                    max_prob = prob
                    predicted_class = cls
            predictions.append(predicted_class)
        return np.array(predictions)

    
    def __repr__(self):
        if self.rules is None:
            return "Prism classifier not fitted yet."
        repr_str = "Prism classifier rules:\n"
        for i, class_rule in enumerate(self.rules):
            repr_str += f"Class {self.classes[i]}: {class_rule}\n"
        return repr_str

d = Dataset()
d.load(filename='numeros.csv')

prism = Prism(alpha=0.5)

X_train, X_test, y_train, y_test = train_test_split(d.X, d.y, test_size=0.2, random_state=2023)

prism.fit(X_train, y_train)

print(prism)
print(prism.predict(X_test))
print(y_test)
