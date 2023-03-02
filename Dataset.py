import numpy as np

class Dataset:
    def __init__(self):
        self.X = None
        self.y = None
        self.feature_names = None
        self.label_name = None

    def load_from_csv(self, filename, label_name=None):
        with open(filename, 'r') as f:
            lines = f.readlines()
        header = lines[0].strip().split(',')
        data = [line.strip().split(',') for line in lines[1:]]
        self.X = np.array(data)[:, :-1]
        self.y = np.array(data)[:, -1]
        self.feature_names = header[:-1]
        self.label_name = header[-1] if label_name is None else label_name

    def load_from_tsv(self, filename, label_name=None):
        with open(filename, 'r') as f:
            lines = f.readlines()
        header = lines[0].strip().split('\t')
        data = [line.strip().split('\t') for line in lines[1:]]
        self.X = np.array(data)[:, :-1]
        self.y = np.array(data)[:, -1]
        self.feature_names = header[:-1]
        self.label_name = header[-1] if label_name is None else label_name

    def describe(self):
        print('Dataset summary:')
        print('----------------')
        print(f'Number of samples: {self.X.shape[0]}')
        print(f'Number of features: {self.X.shape[1]}')
        print(f'Feature names: {self.feature_names}')
        print(f'Label name: {self.label_name}')
        print(f'Label values: {np.unique(self.y)}')

        for i in range(self.X.shape[1]):
            col = self.X[:, i].astype(np.float)
            print(f'Feature "{self.feature_names[i]}":')
            print(f'  Type: {col.dtype}')
            print(f'  Min: {np.min(col)}')
            print(f'  Max: {np.max(col)}')
            print(f'  Mean: {np.mean(col)}')
            print(f'  Std: {np.std(col)}')
            print(f'  Number of missing values: {np.sum(np.isnan(col))}')

    def count_missing_values(self):
        return np.sum(np.isnan(self.X))

    def replace_missing_values(self, method='most_frequent'):
        if method == 'most_frequent':
            # Replace missing values with the most frequent value for each column
            for i in range(self.X.shape[1]):
                # try:
                #     col = self.X[:, i].astype(float)
                # except Exception:
                #     print(f'Warning: column {i} is not numeric, skipping')
                mode = np.nanmode(col)
                col[np.isnan(col)] = mode
                self.X[:, i] = col.astype(str)
        elif method == 'mean':
            # Replace missing values with the mean for each column
            for i in range(self.X.shape[1]):
                try:
                    col = self.X[:, i].astype(float)
                    mean = np.nanmean(col)
                    col[np.isnan(col)] = mean
                    self.X[:, i] = col.astype(str)
                except Exception:
                    print(f'Warning: column {i} is not numeric, skipping')
        else:
            raise ValueError(f'Invalid method "{method}" for replacing missing values')


d = Dataset()
d.load_from_csv('teste.csv')
print(d.X)
print(d.count_missing_values())
