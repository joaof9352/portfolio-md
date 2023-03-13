import numpy as np

class Dataset:
    def __init__(self):
        """
            TODO: Deveríamos guardar os "encodings"?
        """
        self.X = None
        self.y = None
        self.feature_names = None
        self.label_name = None
        self.nums = None
        self.cats = None

    def __read_datatypes(self, filename, sep):
        """
            Lê os tipos da primeira linha do arquivo e armazena em self.nums ou self.cats o numero da coluna.

            TODO: Se a primeira linha tiver um missing value é assumida como categórica ...
        """
        with open(filename) as f:
            header = f.readline().rstrip().split(sep)

            self.feature_names = header[:-1]
            self.label_name = header[-1]
            line = f.readline().rstrip().split(sep)

            self.nums = []
            self.cats = []

            for i in range(len(line)):
                try:
                    float(line[i])
                    self.nums.append(i)
                except ValueError:
                    self.cats.append(i)


    def load(self, filename, sep = ',', label_name=None):
        """
            Carrega o dataset do arquivo filename, separado por sep.
            Feature_names, label_name, X e y são armazenados como atributos da classe.

            TODO: Quando fazemos encoding de dados categóricos, os missing values são considerados como uma categoria.
        """
        self.__read_datatypes(filename, sep)

        numeric_data = np.genfromtxt(filename, delimiter=sep, usecols=self.nums, skip_header=True)
        categorical_data = np.genfromtxt(filename, dtype='U', delimiter=sep, usecols=self.cats, skip_header=True)
        
        # encode categorical_data
        categorical_data = self.label_encode(categorical_data)
        print(f'===============> {numeric_data.shape}')
        print(f'===============> {categorical_data.shape}')
        if numeric_data.ndim == 1:
            numeric_data = numeric_data.reshape(-1, 1)

        if categorical_data.ndim == 1:
            categorical_data = categorical_data.reshape(-1, 1)

        print(f'===============> {numeric_data.shape}')
        print(f'===============> {categorical_data.shape}')

        data = np.empty_like(np.hstack((numeric_data, categorical_data)))
        data[:, self.nums] = numeric_data
        data[:, self.cats] = categorical_data

        self.X = np.array(data)[:, :-1]
        self.y = np.array(data)[:, -1]

        if label_name:
            self.label_name = label_name

    def label_encode(self, arr):
        encoded_arr = np.empty_like(arr)

        if len(arr.shape) == 1:
            unique_vals, encoded_arr[:] = np.unique(arr[:], return_inverse=True)
        else: 
            for i in range(arr.shape[1]):
                unique_vals, encoded_arr[:,i] = np.unique(arr[:,i], return_inverse=True)
        
        return encoded_arr

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
d.load('notas.csv')
print(d.describe())
#print(d.count_missing_values())
