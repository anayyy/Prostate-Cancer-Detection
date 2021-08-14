#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class ProstateCancer:
    def fun(self, re):
        df=pd.read_csv("Prostate_Cancer.csv")
        df = df.drop('id', axis=1)

        for i in df['diagnosis_result']:
            if(i == "M"):
                df['diagnosis_result'] = df['diagnosis_result'].replace("M",1)
            else:
                df['diagnosis_result'] = df['diagnosis_result'].replace("B",0)
    
        y_val= df["diagnosis_result"]
        x_data=df.drop("diagnosis_result",axis=1)


        X_train, X_eval,y_train,y_eval=train_test_split(x_data,y_val,test_size=0.3,random_state=101)

        scaler_model = MinMaxScaler()

        scaler_model.fit(X_train)
        X_train=pd.DataFrame(scaler_model.transform(X_train),columns=X_train.columns,index=X_train.index)

        scaler_model.fit(X_eval)
        X_eval=pd.DataFrame(scaler_model.transform(X_eval),columns=X_eval.columns,index=X_eval.index)

        scaler_model.fit(re)
        re=pd.DataFrame(scaler_model.transform(re),columns=re.columns,index=re.index)

        #Creating Feature Columns
        feat_cols=[]
        for cols in df.columns[1:]:
            column=tf.feature_column.numeric_column(cols)
            feat_cols.append(column)
    
        #The estimator model
        model=tf.compat.v1.estimator.DNNRegressor(hidden_units=[128,256,128],feature_columns=feat_cols)

        #the input function
        input_func=tf.compat.v1.estimator.inputs.pandas_input_fn(X_train,y_train,batch_size=10,num_epochs=1000,shuffle=True)

        model.train(input_fn=input_func,steps=1000)
        
        
        train_metrics=model.evaluate(input_fn=input_func,steps=1000)

        pred_input_func=tf.compat.v1.estimator.inputs.pandas_input_fn(x=re,y=None,batch_size=10,num_epochs=1,shuffle=False)
        preds=model.predict(input_fn=pred_input_func)
            
        predictions=list(preds)
        final_pred=[]
        for pred in predictions:
            final_pred.append(pred["predictions"])
            
        for i in final_pred:
            return(round(i[0]))

from flask import Flask
from flask import render_template
from flask import request
app=Flask(__name__)

app.static_folder='static'

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():

    RESULT = ['CANCER','NO_CANCER']
    
    re = pd.DataFrame(columns=['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness',
       'symmetry', 'fractal_dimension'])
    
    re = re.append({'radius': float(request.form['Radius']), 
                    'texture': float(request.form['Texture']),
                    'perimeter': float(request.form['Perimeter']),
                    'area': float(request.form['Area']),
                    'smoothness': float(request.form['Smoothness']),
                    'compactness': float(request.form['Compactness']),
                    'symmetry': float(request.form['Symmetry']),
                    'fractal_dimension': float(request.form['Fractual_Dimension'])}, ignore_index=True                                        
                    )
    
    
    n = ProstateCancer()
    p = n.fun(re)
    testresult=RESULT[int(p)]

    if(testresult=="CANCER"):
        return render_template('positive.html')
    if(testresult=="NO_CANCER"):
        return render_template('negative.html')
    
    return render_template('resulterror.html')
    





