import streamlit as st 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("heart_disease_data.csv")


X = df.drop(columns='target',axis = 1)
Y = df['target']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1, stratify = Y, random_state = 2)

model = LogisticRegression()
model.fit(X_train, Y_train)

#Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)



# Accuracy on testing data 
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)



# Creating UI elements for data input
st.markdown("<h1 style=align:center;'>Heart Disease AI Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<h3 style=align:center;'>Please enter the data input of your patient </h3>", unsafe_allow_html=True)

age = st.number_input("age", value=0, step=1)
sex = st.selectbox("sex", [0, 1])
cp = st.number_input("cp", value=0, step=1)
trestbps = st.number_input("trestbps", value=0, step=1)
chol = st.number_input("chol", value=0, step=1)
fbs = st.number_input("fbs", value=0, step=1)
restecg = st.number_input("restecg", value=0, step=1)
thalach = st.number_input("thalach", value=0, step=1)
exang = st.number_input("exang", value=0, step=1)
oldpeak = st.number_input("oldpeak", value=0, step=1)
slope = st.number_input("slope", value=0, step=1)
ca = st.number_input("ca", value=0, step=1)
thal = st.number_input("thal", value=0, step=1)



button = st.button("Predict")
if button:
    user_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    prediction = model.predict([user_data])
    if prediction == 0:
       st.markdown("<h4 style=align:center;color=green'>The patient does not have heart issues</45>", unsafe_allow_html=True)
       
    else:
        st.markdown("<h4 style=align:center;color=red'>The patient does not have heart issues</h4>", unsafe_allow_html=True)
        
        st.write(f"Accuracy on training data is {training_data_accuracy}")
        st.write(f"Accuracy on testing data is {testing_data_accuracy}")
        
footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Coyright @Nyanda Jr</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
