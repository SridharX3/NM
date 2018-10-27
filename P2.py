
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


def feature_addition(csv_path):
    img_features = pd.read_csv(csv_path, index_col=0)

    img_features = img_features.assign(key=1).merge(img_features.assign(key=1), on="key", suffixes=["_A", "_B"]).drop("key", axis=1)
    img_features = img_features[img_features["img_id_A"] != img_features["img_id_B"]]
    
    return img_features

def feature_subtraction(csv_path, num_features):
    img_features = pd.read_csv(csv_path, index_col=0)
    # img_features = pd.read_csv(csv_path, index_col=None)    
    
    feature_obj = {}
    feature_obj["img_id"] = "str"
    for i in range(1, num_features+1):
        feature_obj["f"+str(i)] = "uint8"

    img_features = img_features.astype(feature_obj)

    img_features = img_features.assign(key=1).merge(img_features.assign(key=1), on="key", suffixes=["_A", "_B"]).drop("key", axis=1)
    img_features = img_features[img_features["img_id_A"] != img_features["img_id_B"]]


    for idx in range(1, num_features+1):
        img_features["f%d" % idx] = np.abs(img_features["f%d_A" % idx] - img_features["f%d_B" % idx]) 
     
    for label in ["A", "B"]:
        for idx in range(1, num_features+1):
            del img_features["f%d_%s" % (idx, label)]
    
    return img_features


# In[4]:


def read_labeled_data(csv_paths):
    labeled_data = pd.read_csv(csv_paths[0])
    for path in csv_paths[1:]:
        labeled_data = pd.concat([labeled_data, pd.read_csv(path)])
    
    return labeled_data


# In[6]:


def filter_writer_pairs(img_features, labeled_data):
    # filter from all combinations of writer pairs
    a = img_features["img_id_A"] + img_features["img_id_B"]
    b = labeled_data["img_id_A"] + labeled_data["img_id_B"]
    feature_set = img_features[a.isin(b)]
    
    feature_set = pd.merge(feature_set, labeled_data, on=["img_id_A", "img_id_B"])
    
    feature_set["writer_A"] = [elm[:4] for elm in feature_set["img_id_A"]]
    return feature_set


# In[7]:


# feature_set = filter_writer_pairs(img_features, labeled_data)


# In[8]:


def train_val_test_split(feature_set):
    unique = feature_set["writer_A"].unique()
    unique = np.random.permutation(unique)

    # training validation and test sets split
    tr_idx = int(0.8 * unique.shape[0])
    tr_s = unique[:tr_idx]

    training_set = feature_set.loc[feature_set["writer_A"].isin(tr_s)]


    val_idx = tr_idx + int(0.1 * unique.shape[0])
    val_s = unique[tr_idx: val_idx]
    validation_set = feature_set.loc[feature_set["writer_A"].isin(val_s)]


    test_s = unique[val_idx:]
    test_set = feature_set.loc[feature_set["writer_A"].isin(test_s)]
    
    del training_set["writer_A"]
    del validation_set["writer_A"]
    del test_set["writer_A"]
    
    return training_set, validation_set, test_set


# In[9]:


# training_set, validation_set, test_set = train_val_test_split(feature_set)


# In[52]:


# number of features is dynamic here
def get_features_and_labels(dataset, num_features, feature_extraction_type):
    if feature_extraction_type == "addition":
        feature_columns = [ ("f%d_%s" % (idx, label)) for label in ["A", "B"] for idx in range(1, num_features+1) ]
    elif feature_extraction_type == "subtraction":
        feature_columns = [ ("f%d" % idx) for idx in range(1, num_features+1) ]
    
    y_values = dataset["target"]
    dataset = dataset.loc[:, feature_columns]
    dataset.insert(0, "intercept", 1)
    return np.array(dataset, dtype="uint8"), np.array(y_values, dtype="uint8")


# In[53]:


def get_data_split(dataset_type, feature_extraction_type):

#     img_features = feature_addition("./HumanObserved-Dataset/HumanObserved-Features-Data/HumanObserved-Features-Data.csv", 9)
    
    if dataset_type == "human":
        csv_features_path = "./HumanObserved-Dataset/HumanObserved-Features-Data/HumanObserved-Features-Data.csv"
        csv_paths = ["./HumanObserved-Dataset/HumanObserved-Features-Data/diffn_pairs.csv", "./HumanObserved-Dataset/HumanObserved-Features-Data/same_pairs.csv"]
        num_features = 9
    elif dataset_type == "gsc":
        csv_features_path = "./GSC-Dataset/GSC-Features-Data/GSC-Features.csv"
        csv_paths = ["./GSC-Dataset/GSC-Features-Data/diffn_pairs.csv", "./GSC-Dataset/GSC-Features-Data/same_pairs.csv"]
        num_features = 512
        
    if feature_extraction_type == "addition":
        img_features = feature_addition(csv_features_path)
    elif feature_extraction_type == "subtraction":
        img_features = feature_subtraction(csv_features_path, num_features)

    labeled_data = read_labeled_data(csv_paths)
    
    filtered_feature_set = filter_writer_pairs(img_features, labeled_data)
    
    training_set, validation_set, test_set = train_val_test_split(filtered_feature_set)
    
    X_train, y_train = get_features_and_labels(training_set, num_features=num_features, feature_extraction_type=feature_extraction_type)
    X_val, y_val = get_features_and_labels(validation_set, num_features=num_features, feature_extraction_type=feature_extraction_type)
    X_test, y_test = get_features_and_labels(test_set, num_features=num_features, feature_extraction_type=feature_extraction_type)

    return X_train, y_train, X_val, y_val, X_test, y_test


# print(loss)


# In[56]:


def calculate_logistic_loss(X_train, y_train, W):
    
    loss = 0
    prd = np.dot(X_train, W)
    h_x = 1 / (1 + np.exp(-prd))
    a = np.dot(-np.log(h_x).T, y_train)
    b = np.dot(-np.log(1 - h_x).T, (1 - y_train))
    loss = (np.sum(a+b))
    
    loss =  loss / (X_train.shape[0])
        
    return loss


# In[58]:


X_train, y_train, X_val, y_val, X_test, y_test = get_data_split(dataset_type="human", feature_extraction_type="addition")
# X_train, y_train, X_val, y_val, X_test, y_test = get_data_split(dataset_type="gsc", feature_extraction_type="subtraction")
# X_train, y_train, X_val, y_val, X_test, y_test = get_data_split(dataset_type="human", feature_extraction_type="subtraction")
W = np.zeros((X_train.shape[1], 1))

loss = calculate_logistic_loss(X_train, y_train, W)
print(loss)


# In[59]:


def gradient_descent(W, X_train,y_train, learning_rate=0.1):
    
    for k in range(200):
        if k % 10 == 0:
            print("Iteration %d" % k)        
        
        prd = np.dot(X_train, W)
        h_x = 1 / (1 + np.exp(-prd)) 
        
        y_train = y_train.reshape((y_train.shape[0], 1))
        np.subtract(h_x, y_train, out=h_x)
        h_x = h_x.T  
        v = np.dot(h_x, X_train)
        gradient = np.sum(v)
        # print(gradient)
        
        W = W - (learning_rate * gradient / X_train.shape[0])
    return W


# In[60]:


W = gradient_descent(W, X_train, y_train)
print(W)

# In[61]:


diff = 0

prd = np.dot(X_val, W)
h_x = 1 / (1 + np.exp(-prd)) 

h_x[h_x >= 0.5] = 1
h_x[h_x < 0.5] = 0


y_val = y_val.reshape((y_val.shape[0], 1))
np.abs(np.subtract(h_x, y_val, out=h_x), out=h_x)

diff = np.sum(h_x)

accuracy = 1 - (diff / X_val.shape[0])
print(accuracy)


prd = np.dot(X_test, W)
h_x = 1 / (1 + np.exp(-prd)) 

h_x[h_x >= 0.5] = 1
h_x[h_x < 0.5] = 0


y_test = y_test.reshape((y_test.shape[0], 1))
np.abs(np.subtract(h_x, y_test, out=h_x), out=h_x)

diff = np.sum(h_x)

accuracy = 1 - (diff / X_test.shape[0])
print(accuracy)