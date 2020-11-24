from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
app = Flask(__name__)


@app.route('/', methods=['POST'])
def predict():
    json_ = request.get_json()
    print(json_)
    df = pd.DataFrame([json_], dtype=float)
    print(df)
    
    # Data processing
    to_change = ['PTS', 'FGM', 'FGA', '3P Made', '3PA', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK','TOV']
    df.fillna(0,inplace=True)
    for col in to_change:
        df[col] = df[col]/df['MIN']
    df['EFF'] = df['PTS'] + df['REB'] + df['AST'] + df['STL'] + df['BLK'] - (df['FGA']-df['FGM']) - (df['FTA']-df['FTM']) - df['TOV']
    for col in ['DREB','REB','TOV']:
        df[col]=np.log(df[col])
    for col in ['OREB','AST','STL','BLK']:
        df[col]=np.sqrt(df[col])
    scl = joblib.load('scaler.joblib')
    X = scl.transform(df.drop(['Name'],axis=1).values)
    
    # Predicting
    clf = joblib.load('classifier.joblib')
    prediction = clf.predict(X)
    return {'prediction':list(prediction)}

if __name__ == '__main__':
    clf = joblib.load('classifier.joblib')
    scl = joblib.load('scaler.joblib')
    app.run(port=8080)