
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("medical-charges.csv")

def split_data(df):
    features = ["age", "sex", "bmi", "children", "smoker", "region"]
    y = df["charges"]
    X = df[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    training_data = pd.concat([X_train, y_train], axis=1)
    testing_data = pd.concat([X_test, y_test], axis=1)
    return training_data, testing_data

def onehotEncode(df, categ_to_encode, normal_categ):
    enc = OneHotEncoder(drop='first', sparse_output=False)
    encoded_categ = enc.fit_transform(df[categ_to_encode])
    
    encoded_categ = pd.DataFrame(
        encoded_categ, 
        columns=enc.get_feature_names_out(categ_to_encode),
        index=df.index)
    encoded_combined_df = pd.concat([encoded_categ, df[normal_categ]], axis=1)

    return encoded_combined_df, enc

def standardScale(df, categ_to_scale):
    scale = StandardScaler()
    scaled_categ = scale.fit_transform(df[categ_to_scale])
    df[categ_to_scale] = scaled_categ
    return df, scale

def Preprocess(current_data, enc=None, scale=None):
    """ Normalize, Encode, Feature Engineering """
    current_data = current_data.copy()
    current_data["age_bmi"] = current_data["age"] * current_data["bmi"]

    categ_to_encode = ["sex", "smoker", "region"]
    normal_categ = ["age", "bmi", "children", "charges", "age_bmi"]
    categ_to_scale = ["age", "bmi", "charges", "age_bmi"]

    # Encode categorical
    if enc is None:
        encoded_df, enc = onehotEncode(current_data, categ_to_encode, normal_categ)
    else:
        transformed = enc.transform(current_data[categ_to_encode])
        encoded_df = pd.DataFrame(
            transformed,
            columns=enc.get_feature_names_out(categ_to_encode),
            index=current_data.index
        )
        encoded_df = pd.concat([encoded_df, current_data[normal_categ]], axis=1)

    if scale is None:
        scaled_df, scale = standardScale(encoded_df, categ_to_scale)
    else:
        scaled_numeric = scale.transform(encoded_df[categ_to_scale])
        scaled_numeric = pd.DataFrame(scaled_numeric, columns=categ_to_scale, index=encoded_df.index)
        cat_cols = [c for c in encoded_df.columns if c not in categ_to_scale]
        scaled_df = pd.concat([encoded_df[cat_cols], scaled_numeric], axis=1)

    return scaled_df, enc, scale


def drop_features(scaled_df, tolerance=0.3):
    corr = scaled_df.corr()["charges"]
    final_features = corr[abs(corr) >= tolerance].index
    return scaled_df[final_features], final_features


def gradients(x, y, yhat):
    m = len(y)
    dw = (-2/m) * np.dot(x.T, (y-yhat))
    db = (-2/m) * np.sum(y-yhat)
    return dw, db

def gradient_descent(w, b, dw, db, learning_rate):
    temp_w = w - (dw*learning_rate)
    temp_b = b - (db*learning_rate)
    w, b = temp_w, temp_b
    return w, b

def train(x, y, learning_rate, epoch, tolerance):
    w, b = np.zeros(x.shape[1]), 0
    for i in range(epoch):
        prev_w, prev_b = w, b
        yhat = np.dot(x, w) + b
        dw, db = gradients(x, y, yhat)
        w, b = gradient_descent(w, b, dw, db, learning_rate)

        if np.linalg.norm(w - prev_w) < tolerance and abs(b - prev_b) < tolerance:
            break


        if (i%100==0):
            print(f"Current Epoch: {i}, Current Weight: {w}, Current Bias: {b}")
    return w, b

def extract_features(final_df):
    y = final_df["charges"]
    final_df = final_df.drop(columns="charges")
    x = final_df
    return x, y

def Regress(final_df, learning_rate=0.01, epoch=10000, tolerance=1e-6):
    x, y = extract_features(final_df)
    w, b = train(x, y, learning_rate, epoch, tolerance)
    return w, b

def Predict(w, b, testing_data, scale, enc, final_features):
    scaled_df, _, _ = Preprocess(testing_data, enc=enc, scale=scale)
    final_df = scaled_df[final_features]
    x, y = extract_features(final_df)
    yhat_scaled = np.dot(x, w) + b
    yhat = scale.inverse_transform(yhat_scaled.reshape(-1, 1)).flatten()
    original_y = scale.inverse_transform(y.values.reshape(-1, 1)).flatten()
    mae = np.mean(np.abs(original_y - yhat))
    return f"{mae}"


def main():
    training_data, testing_data = split_data(df)
    scaled_df, enc, scale = Preprocess(training_data)
    print(scaled_df.corr()["charges"])
    final_df, final_features = drop_features(scaled_df)
    print(final_df)
    w, b = Regress(final_df)
    mae = Predict(w, b, testing_data, scale, enc, final_features)
    print(mae)



main()