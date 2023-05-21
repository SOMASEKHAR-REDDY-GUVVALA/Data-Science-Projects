
"""
fakebills_classification logistic model deployement

"""

import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression

st.title('Model Deployment: Logistic Regression')

st.sidebar.header('User Input Parameters')

def user_input_features():
    diagonal = st.sidebar.number_input('enter diagonal value')
    height_left = st.sidebar.number_input('enter height_left value')
    height_right = st.sidebar.number_input('enter height_right value')
    margin_low = st.sidebar.number_input("enter margin_low value")
    margin_up = st.sidebar.number_input("enter margin_up value")
    length = st.sidebar.number_input('enter length value')
    
    data = {'diagonal':diagonal,
            'height_left':height_left,
            'height_right':height_right,
            'margin_low':margin_low,
            'margin_up':margin_up,
           'length':length}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

fake_bills = pd.read_csv("fake_bills.csv",sep=';')

fakebills_lab=fake_bills.copy()

from sklearn import preprocessing

label_encoder=preprocessing.LabelEncoder()
fakebills_lab['is_genuine']=label_encoder.fit_transform(fakebills_lab['is_genuine'])
train = fakebills_lab[fakebills_lab['margin_low'].notna()].copy()
test = fakebills_lab[fakebills_lab['margin_low'].isna()].copy()
X_train=train.drop("margin_low",axis=1)
y_train=train.margin_low
X_test=test.drop("margin_low",axis=1)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

test['margin_low'] = y_pred
fakebills_nona = pd.concat([train, test])
fakebills_nona=fakebills_nona.reset_index()
fakebills_nona=fakebills_nona.drop('index',axis=1)

# dv and iv
X=fakebills_nona.drop('is_genuine',axis=1)
Y=fakebills_nona['is_genuine']


clf = LogisticRegression(C= 78.47599703514607,max_iter= 500,penalty= 'l1',solver='liblinear')
clf.fit(X,Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Predicted Result')
st.write('genuine invoice' if prediction_proba[0][1] > 0.5 else 'fake invoice')

st.subheader('Prediction Probability')
st.write(prediction_proba)