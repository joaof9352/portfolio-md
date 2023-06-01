import numpy as np
from Dataset import Dataset

class DecisionTree:
    def __init__(self, attribute_selection='entropy', pruning=None):
        self.attribute_selection = attribute_selection
        self.pruning = pruning
        self.tree = None

    def fit(self, X, y):
        attributes = np.arange(X.shape[1])  # indices das colunas de atributos
        self.tree = self._build_tree(X, y, attributes)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for i, instance in enumerate(X):
            predictions[i] = self._traverse_tree(instance, self.tree)
        return predictions

    def __repr__(self):
        return self._print_tree(self.tree)

    def _build_tree(self, X, y, attributes):
        if len(np.unique(y)) == 1:  # todos os exemplos possuem a mesma classe
            return {'class': y[0]}

        if len(attributes) == 0:  # não há mais atributos para dividir
            unique_classes, class_counts = np.unique(y, return_counts=True)
            majority_class = unique_classes[np.argmax(class_counts)]
            return {'class': majority_class}

        # Seleção do atributo de acordo com o critério escolhido
        if self.attribute_selection == 'entropy':
            selected_attribute = self._select_attribute_entropy(X, y, attributes)
        elif self.attribute_selection == 'gini':
            selected_attribute = self._select_attribute_gini(X, y, attributes)
        elif self.attribute_selection == 'gain_ratio':
            selected_attribute = self._select_attribute_gain_ratio(X, y, attributes)
        else:
            raise ValueError("Invalid attribute selection method.")

        tree = {'attribute': selected_attribute, 'children': {}}

        # Construção dos filhos
        attribute_values = np.unique(X[:, selected_attribute])
        for value in attribute_values:
            subset_mask = X[:, selected_attribute] == value
            subset_X = X[subset_mask]
            subset_y = y[subset_mask]
            if subset_X.shape[0] == 0:  # nenhum exemplo com esse valor do atributo
                unique_classes, class_counts = np.unique(y, return_counts=True)
                majority_class = unique_classes[np.argmax(class_counts)]
                tree['children'][value] = {'class': majority_class}
            else:
                remaining_attributes = np.delete(attributes, selected_attribute)
                tree['children'][value] = self._build_tree(subset_X, subset_y, remaining_attributes)

        return tree

    def _select_attribute_entropy(self, X, y, attributes):
        entropy = self._calculate_entropy(y)
        information_gains = np.zeros(len(attributes))
        for i, attribute in enumerate(attributes):
            attribute_values = np.unique(X[:, attribute])
            subset_entropies = np.zeros(len(attribute_values))
            for j, value in enumerate(attribute_values):
                subset_mask = X[:, attribute] == value
                subset_y = y[subset_mask]
                subset_entropies[j] = self._calculate_entropy(subset_y)
            subset_sizes = np.array([len(X[X[:, attribute] == value]) for value in attribute_values])
            information_gains[i] = entropy - np.sum(subset_entropies * subset_sizes) / len(y)
        return attributes[np.argmax(information_gains)]

    def _calculate_entropy(self, y):
        _, class_counts = np.unique(y, return_counts=True)
        probabilities = class_counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def _select_attribute_gini(self, X, y, attributes):
        gini_index = self._calculate_gini_index(y)
        gini_indices = np.zeros(len(attributes))
        for i, attribute in enumerate(attributes):
            attribute_values = np.unique(X[:, attribute])
            subset_ginis = np.zeros(len(attribute_values))
            for j, value in enumerate(attribute_values):
                subset_mask = X[:, attribute] == value
                subset_y = y[subset_mask]
                subset_ginis[j] = self._calculate_gini_index(subset_y)
            subset_sizes = np.array([len(X[X[:, attribute] == value]) for value in attribute_values])
            gini_indices[i] = np.sum(subset_ginis * subset_sizes) / len(y)
        return attributes[np.argmin(gini_indices)]

    def _calculate_gini_index(self, y):
        _, class_counts = np.unique(y, return_counts=True)
        probabilities = class_counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _select_attribute_gain_ratio(self, X, y, attributes):
        selected_attribute = self._select_attribute_entropy(X, y, attributes)
        attribute_values = np.unique(X[:, selected_attribute])
        subset_ratios = np.zeros(len(attribute_values))
        for i, value in enumerate(attribute_values):
            subset_mask = X[:, selected_attribute] == value
            subset_y = y[subset_mask]
            subset_ratios[i] = len(subset_y) / len(y)
        gain_ratios = np.zeros(len(attributes))
        for i, attribute in enumerate(attributes):
            if attribute == selected_attribute:
                continue
            attribute_values = np.unique(X[:, attribute])
            subset_entropies = np.zeros(len(attribute_values))
            for j, value in enumerate(attribute_values):
                subset_mask = X[:, attribute] == value
                subset_y = y[subset_mask]
                subset_entropies[j] = self._calculate_entropy(subset_y)
            subset_sizes = np.array([len(X[X[:, attribute] == value]) for value in attribute_values])
            gain_ratios[i] = self._calculate_entropy(y) - np.sum(subset_entropies * subset_sizes) / len(y)
        return attributes[np.argmax(gain_ratios * subset_ratios)]

    def _traverse_tree(self, instance, node):
        if 'class' in node:
            return node['class']
        attribute = node['attribute']
        value = instance[attribute]
        if value not in node['children']:
            return None  # valor do atributo não presente na árvore (tratamento de conflito)
        return self._traverse_tree(instance, node['children'][value])

    def _print_tree(self, node, indent=''):
        if 'class' in node:
            return str(node['class'])
        attribute = node['attribute']
        tree_str = ''
        for value, child in node['children'].items():
            tree_str += f'{indent}{attribute} = {value} -> {self._print_tree(child, indent + "  ")}\n'
        return tree_str

d = Dataset()
d.load(filename='notas.csv')
#print(_entropy(d.X, d.y))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(d.X, d.y, test_size=0.2, random_state=42)

tree = DecisionTree(attribute_selection='entropy')

# Treinando a árvore de decisão
tree.fit(X_train, y_train)

# Imprimindo a árvore
print(tree)

from sklearn.metrics import accuracy_score

# Fazendo previsões no conjunto de teste
y_pred = tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy}')

