import numpy as np

class Dataset:
    def __init__(self):
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

        numeric_data = np.genfromtxt(filename, delimiter=sep, usecols=self.nums, skip_header=True)#, filling_values=np.nan, missing_values='')
        categorical_data = np.genfromtxt(filename, delimiter=sep, dtype='U32', usecols=self.cats, skip_header=True, missing_values='')#, filling_values=np.nan, )
        

        if numeric_data.ndim == 1:
            numeric_data = numeric_data.reshape(-1, 1)

        if categorical_data.ndim == 1:
            categorical_data = categorical_data.reshape(-1, 1)
        
        if len(self.cats) > 0:
            
            #Label Encoder
            categorical_data, encoding_dict = self.label_encode(categorical_data)
        
            data = np.empty_like(np.hstack((numeric_data, categorical_data)))
            data[:, self.nums] = numeric_data
            data[:, self.cats] = categorical_data
        else:
            data = numeric_data

        self.X = np.array(data)[:, :-1]
        self.y = np.array(data)[:, -1]

        print(self.X)

        if label_name:
            self.label_name = label_name

    def label_encode(self, arr):
        encoded_arr = np.empty_like(arr)
        encoding_dicts = []
        encoding_dict = {}

        if len(arr.shape) == 1:
            unique_vals, encoded_arr[:] = np.unique(arr[:], return_inverse=True)
            #missing_indices = np.where(arr[:] == '')[0]
            #encoded_arr[:] = np.where(missing_indices, np.nan, encoded_arr[:])
            for i in range(len(unique_vals)):
                encoding_dict[unique_vals[i]] = i
            encoding_dicts.append(encoding_dict)
        else: 
            for i in range(arr.shape[1]):
                unique_vals, encoded_arr[:,i] = np.unique(arr[:,i], return_inverse=True)
                # missing_indices = np.where(arr[:,i] == '')[0]
                # encoded_arr[:,i] = np.where(missing_indices, np.nan, encoded_arr[:,i])
                for j in range(len(unique_vals)):
                    encoding_dict[unique_vals[j]] = j
                encoding_dicts.append(encoding_dict)
                encoding_dict = {}

        return encoded_arr, encoding_dicts

    def dropna(self):
        """
        Remove rows with missing values from the dataset.
        """
        # Find indices of rows with missing numeric values
        numeric_missing = np.isnan(self.X.astype(float)).any(axis=1)

        # Find indices of rows with missing string values
        string_missing = np.char.strip(self.X.astype(str)) == ''
        string_missing = string_missing.any(axis=1)

        # Combine missing indices
        missing_indices = numeric_missing | string_missing

        # Filter X and y based on the indices
        self.X = self.X[~missing_indices]
        self.y = self.y[~missing_indices]

    def describe(self):
        print('Dataset summary:')
        print('----------------')
        print(f'Number of samples: {self.X.shape[0]}')
        print(f'Number of features: {self.X.shape[1]}')
        print(f'Feature names: {self.feature_names}')
        print(f'Label name: {self.label_name}')
        print(f'Label values: {np.unique(self.y)}')

        for i in range(self.X.shape[1]):
            col = self.X[:, i].astype(float)
            print(f'Feature "{self.feature_names[i]}":')
            print(f'  Type: {col.dtype}')
            print(f'  Min: {np.nanmin(col)}')
            print(f'  Max: {np.nanmax(col)}')
            print(f'  Mean: {np.nanmean(col)}')
            print(f'  Std: {np.nanstd(col)}')
            print(f'  Number of missing values: {np.sum(np.isnan(col))}')

    def train_test_split(self, test_size=0.2, random_state=None):
        """
        Divide o conjunto de dados em conjuntos de treinamento e teste.

        Parâmetros:
        X: array de features
        y: array de saídas
        test_size: float, proporção do conjunto de teste
        random_state: int, semente para o gerador de números aleatórios

        Retorna:
        X_train: array de features de treinamento
        X_test: array de features de teste
        y_train: array de saídas de treinamento
        y_test: array de saídas de teste
        """

        if random_state:
            np.random.seed(random_state)

        n_samples = self.X.shape[0]
        n_test = int(n_samples * test_size)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        X_train = self.X[train_indices]
        y_train = self.y[train_indices]
        X_test = self.X[test_indices]
        y_test = self.y[test_indices]

        return X_train, X_test, y_train, y_test

    def count_missing_values(self):
        return np.sum(np.isnan(self.X), axis=0)

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