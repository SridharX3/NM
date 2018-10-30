import pandas as pd
import numpy as np


def extract(csv_features_path, pair_csv_paths, feature_extraction_type):
    features = pd.read_csv(csv_features_path, index_col="img_id")
    diff_pairs = pd.read_csv(pair_csv_paths[0], index_col=False)
    same_pairs = pd.read_csv(pair_csv_paths[1],index_col=False)

    if "HumanObserved" in csv_features_path:
        del features[(features.columns.get_values())[0]]

    same_pairs = pd.read_csv(pair_csv_paths[0])
    diff_pairs = pd.read_csv(pair_csv_paths[1])

    same_pairs_len = same_pairs.shape[0]
    diff_pairs_len = diff_pairs.shape[0]

    frac = same_pairs_len / diff_pairs_len

    diff_pairs = diff_pairs.sample(frac=frac)

    inputs = np.concatenate((diff_pairs, same_pairs))
    
    if feature_extraction_type == "concatenation":
        if "HumanObserved" not in csv_features_path:
            print("GSC concatenation takes a minute or two")
        
        processed_data = feature_concatenation(inputs, features)
    elif feature_extraction_type == "subtraction":
        if "HumanObserved" not in csv_features_path:
            print("GSC subtraction takes a minute or two")
        
        processed_data = feature_subtraction(inputs, features)
    # now data contains image pairs, features and target
    processed_data = np.hstack((processed_data, inputs[:, -1].reshape(inputs.shape[0], 1)))
    return processed_data
    
def feature_concatenation(inputs, features):
    print("Feature concatenation in progress")
    processed_data = np.empty((inputs.shape[0], features.shape[1]*2+2), dtype="O")

    for idx, row in enumerate(inputs):
        img_id_A = row[0]
        img_id_B = row[1]

        processed_data[idx] = np.array([img_id_A, img_id_B, *features.loc[img_id_A], *features.loc[img_id_B]])

    return processed_data

def feature_subtraction(inputs, features):
    print("Feature subtraction in progress")
    processed_data = np.empty((inputs.shape[0], features.shape[1]+2), dtype="O")

    for idx, row in enumerate(inputs):
        img_id_A = row[0]
        img_id_B = row[1]

        processed_data[idx] = np.array([img_id_A, img_id_B, *np.abs(np.subtract(features.loc[img_id_A], features.loc[img_id_B]))])

    return processed_data


def process_data(dataset_type, feature_extraction_type):

    if dataset_type == "human":
        csv_features_path = "./HumanObserved-Dataset/HumanObserved-Features-Data/HumanObserved-Features-Data.csv"
        pair_csv_paths = ["./HumanObserved-Dataset/HumanObserved-Features-Data/same_pairs.csv", "./HumanObserved-Dataset/HumanObserved-Features-Data/diffn_pairs.csv"]
    elif dataset_type == "gsc":
        csv_features_path = "./GSC-Dataset/GSC-Features-Data/GSC-Features.csv"
        pair_csv_paths = ["./GSC-Dataset/GSC-Features-Data/same_pairs.csv", "./GSC-Dataset/GSC-Features-Data/diffn_pairs.csv"]

    processed_data = extract(csv_features_path, pair_csv_paths, feature_extraction_type)

    print("Writing  to ./%s_%s_processed.csv" % (dataset_type, feature_extraction_type))
    pd.DataFrame(processed_data).to_csv("./%s_%s_processed.csv" % (dataset_type, feature_extraction_type), header=False, index=False)

    return processed_data

# processed_data = process_data("gsc", "concatenation")
# processed_data = process_data("gsc", "subtraction")

processed_data = process_data("human", "subtraction")
# processed_data = process_data("human", "concatenation")