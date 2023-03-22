import pandas as pd
from sklearn.compose import make_column_transformer, make_column_selector
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
import warnings 

def mask(X, sensitive_features, sensitive_values=None):
    X_ = X.copy()
    
    sensitive = np.zeros(len(X_)).astype(bool)

    if sensitive_values is None:
        sensitive_values = np.ones_like(sensitive_features).astype(bool)
        
    for f, v in zip(sensitive_features, sensitive_values):
        sensitive = sensitive | (X_[f] == v)
        
    return sensitive

def one_hot(data, cat_features, columns):
        enc = make_column_transformer((OneHotEncoder(drop='if_binary', sparse=False), cat_features), remainder="drop")
        df_cat_array = enc.fit_transform(data)
        
        was_encoded = enc.transformers_[0][1].drop_idx_ == None
        i_encoded = np.where(was_encoded)[0]
        
        new_columns = []
        new_cat_features = []
        cat_encoded_features = []
        cat_not_encoded_features = []

        for feature in columns:
            if feature in enc.transformers_[0][2]:
                i = enc.transformers_[0][2].index(feature)
                if i in i_encoded:
                    new_features = enc.transformers_[0][1].categories_[i]
                    new_columns.extend(list(new_features))
                    new_cat_features.extend(list(new_features))
                    cat_encoded_features.extend(list(new_features))
                else:
                   new_columns.append(feature)
                   new_cat_features.append(feature)
                   cat_not_encoded_features.append(feature)
            else:
                new_columns.append(feature)
        
        columns = new_columns
        cat_features = new_cat_features

        return columns, cat_features, df_cat_array

def load_kdd(csv_path):
    cat_columns = [" industry code", " occupation code", " own business or self employed"]
    target_label = " income"
    s_features = [" sex"]
    s_values = [0]
    #na_vals = ['?']

    data = pd.read_csv(csv_path)
    #for v in na_vals:
        #data.replace(v, np.nan, inplace=True)
    data.dropna(inplace=True)

    data = data.astype({label: object for label in cat_columns})

    index, columns = data.index, data.columns

    num_features = make_column_selector(dtype_include=np.number)(data)
    cat_features = make_column_selector(dtype_include=object)(data)

    scaler = make_column_transformer((StandardScaler(), num_features),remainder="drop")
    data_num_array = scaler.fit_transform(data)

    enc_ordinal = make_column_transformer((OrdinalEncoder(), cat_features), remainder="drop") 
    data_cat_array = enc_ordinal.fit_transform(data)

    data = pd.DataFrame(index=index, columns=columns)
    try:
        data[num_features] = data_num_array
    except e:
        warnings.warn("No operation was done on numerical features.")
    try:
        data[cat_features] = data_cat_array
    except e:
        warnings.warn("No encoding was done on categorical features.")
    
    for f in columns:
        if f in num_features:
            data[f] = data[f].astype(np.float64)
        elif f in cat_features:
            data[f] = data[f].astype(np.int64)
            
    data = data

    y = data.pop(target_label) 
    X = data.copy()
    del data
    
    s_col = mask(X,  s_features, s_values)

    return X, y, s_col

def load_compas(csv_path):
    to_keep = ["sex", "age_cat", "race", "priors_count", "c_charge_degree"]
    cat_columns = ["two_year_recid"]
    target_label = "two_year_recid"
    s_features = ["sex"]
    s_values = [0]

    data = pd.read_csv(csv_path, index_col= 0)


    values = ["African-American", "Caucasian"]
    data = data.loc[data["race"].isin(values)]

    data = data[to_keep + [target_label]]
    data.dropna(inplace=True)


    index, columns = data.index, data.columns

    data = data.astype({label: object for label in cat_columns})

    num_features = make_column_selector(dtype_include=np.number)(data)
    cat_features = make_column_selector(dtype_include=object)(data)

    scaler = make_column_transformer((StandardScaler(), num_features),remainder="drop")
    data_num_array = scaler.fit_transform(data)

    columns, cat_features, data_cat_array = one_hot(data, cat_features, columns)

    data = pd.DataFrame(index=index, columns=columns)
    try:
        data[num_features] = data_num_array
    except e:
        warnings.warn("No operation was done on numerical features.")
    try:
        data[cat_features] = data_cat_array
    except e:
        warnings.warn("No encoding was done on categorical features.")
    
    for f in columns:
        if f in num_features:
            data[f] = data[f].astype(np.float64)
        elif f in cat_features:
            data[f] = data[f].astype(np.int64)
            
    data = data

    y = data.pop(target_label) 
    X = data.copy()
    del data
    
    s_col = mask(X,  s_features, s_values)

    return X, y, s_col

def load_adult_census(csv_path):
    target_label = "income"
    s_features = ["sex"]
    s_values = [1]
    na_vals = ['?']

    data = pd.read_csv(csv_path, sep= ", ", engine="python")
    for v in na_vals:
        data.replace(v, np.nan, inplace=True)
        data.dropna(inplace=True)

    index, columns = data.index, data.columns

    num_features = make_column_selector(dtype_include=np.number)(data)
    cat_features = make_column_selector(dtype_include=object)(data)

    scaler = make_column_transformer((StandardScaler(), num_features),remainder="drop")
    data_num_array = scaler.fit_transform(data)

    columns, cat_features, data_cat_array = one_hot(data, cat_features, columns)

    data = pd.DataFrame(index=index, columns=columns)
    try:
        data[num_features] = data_num_array
    except e:
        warnings.warn("No operation was done on numerical features.")
    try:
        data[cat_features] = data_cat_array
    except e:
        warnings.warn("No encoding was done on categorical features.")
    
    for f in columns:
        if f in num_features:
            data[f] = data[f].astype(np.float64)
        elif f in cat_features:
            data[f] = data[f].astype(np.int64)
            
    data = data

    y = data.pop(target_label) 
    X = data.copy()
    del data
    
    s_col = mask(X,  s_features, s_values)

    return X, y, s_col

def load_bank(csv_path):
    target_label = "y"
    s_features = ["married"]
    s_values = [1]
    na_vals = ['unknown']

    data = pd.read_csv(csv_path, sep= ";")
    for v in na_vals:
        data.replace(v, np.nan, inplace=True)
        data.dropna(inplace=True)

    data = data.assign(contacted=pd.Series(["yes"]*len(data)).values)
    data.loc[data["pdays"] == 999, "contacted"] = "no"
    #print(data["contacted"])

    index, columns = data.index, data.columns

    num_features = make_column_selector(dtype_include=np.number)(data)
    cat_features = make_column_selector(dtype_include=object)(data)

    scaler = make_column_transformer((StandardScaler(), num_features),remainder="drop")
    data_num_array = scaler.fit_transform(data)

    columns, cat_features, data_cat_array = one_hot(data, cat_features, columns)

    data = pd.DataFrame(index=index, columns=columns)
    try:
        data[num_features] = data_num_array
    except e:
        warnings.warn("No operation was done on numerical features.")
    try:
        data[cat_features] = data_cat_array
    except e:
        warnings.warn("No encoding was done on categorical features.")
    
    for f in columns:
        if f in num_features:
            data[f] = data[f].astype(np.float64)
        elif f in cat_features:
            data[f] = data[f].astype(np.int64)
            
    data = data

    y = data.pop(target_label) 
    X = data.copy()
    del data
    
    s_col = mask(X,  s_features, s_values)

    return X, y, s_col
