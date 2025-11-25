import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_FILE="model.pkl"


if not os.path.exists(MODEL_FILE):
    data=pd.read_csv("synthetic_fraud_dataset.csv")
    y=data['is_fraud']
    X=data.drop('is_fraud', axis=1)
    sss=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index, test_index in sss.split(X, y):
        X_train= X.iloc[train_index]
        X_test=X.iloc[test_index].to_csv("input.csv")
        y_train = y.iloc[train_index]
        y_test= y.iloc[test_index].to_csv("actual.csv")
    training=X_train.copy()
    tr_num=training.drop(['transaction_id','user_id','transaction_type','country','merchant_category'],axis=1)
    scaler = StandardScaler()
    train_num = scaler.fit_transform(tr_num)

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000) # Increase max_iter if convergence warnings occur
    model.fit(train_num, y_train)
    joblib.dump(model,MODEL_FILE)
    print("model trained succesfully")
else:
    model=joblib.load(MODEL_FILE)
    input_data=pd.read_csv("input.csv")
    real=pd.read_csv("actual.csv")
    real_val= real.iloc[:, 1:]
    t_num=input_data.drop(['transaction_id','user_id','transaction_type','country','merchant_category'],axis=1)
    df= t_num.iloc[:, 1:]
    scaler = StandardScaler()
    ts_num = scaler.fit_transform(df)

    predictions=model.predict(ts_num)
    input_data["is_fraud"]=predictions
    input_data.to_csv("output.csv",index=False)
    accuracy = accuracy_score(real_val, predictions)
    print(f"accuracy is {accuracy*100}%")