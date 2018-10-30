import numpy as np
import pandas as pd

def get_train_val_test_split(X):
    print("Splitting data into training, validation and test sets")

    X = np.array([np.append(elm, elm[0][:4]) for elm in X])

    unique_writers = np.unique(X[:,-1])
    
    unique_writers = np.random.permutation(unique_writers)

    train_split_idx = int(0.8 * unique_writers.shape[0])
    train_writers = unique_writers[:train_split_idx]


    X_train = X[np.isin(X[:, -1], train_writers)]
    y_train = X_train[:, -2]
    X_train = X_train[:, 2:-2]
    

    val_split_idx = train_split_idx + int(0.1 * unique_writers.shape[0])
    val_writers = unique_writers[train_split_idx: val_split_idx]
    
    X_val = X[np.isin(X[:, -1], val_writers)]
    y_val = X_val[:, -2]
    X_val = X_val[:, 2:-2]

    test_writers = unique_writers[val_split_idx:]
    X_test = X[np.isin(X[:, -1], test_writers)]
    y_test = X_test[:, -2]
    X_test = X_test[:, 2:-2]
    
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def get_data_split(dataset_type, feature_extraction_type):

    print("Reading data from ./%s_%s_processed.csv" % (dataset_type, feature_extraction_type))
    input_data = pd.read_csv("./%s_%s_processed.csv" % (dataset_type, feature_extraction_type), index_col=False)

    input_data = np.array(input_data)

    return get_train_val_test_split(input_data)
