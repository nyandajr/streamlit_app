import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

heart_data = pd.read_csv("heart_disease_data.csv")

# Split data into features and target
X = heart_data.drop(columns='target',axis=1)
Y = heart_data['target']

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y,random_state=2)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Get accuracy of model on training and test data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# Create UI elements for inputting data and displaying results
st.markdown("<h1 style='text-align:center;'>Heart Disease AI Prediction App</h1>", unsafe_allow_html=True)



# heart_img = "heart.webp"
# st.image(heart_img, use_column_width=300)

input_data = st.sidebar.text_input('''Enter patient data in the following order--> 
                        age	,sex, cp,	trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal:''')

if input_data:
    if len(input_data.split(',')) != 13:
        st.markdown("<h2 style='text-align:center;color:red;'>Please enter 13 columns of data only.</h2>", unsafe_allow_html=True)

    else:
        input_data_as_numpy_array = np.asarray(list(map(float, input_data.split(','))))
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = model.predict(input_data_reshaped)

        st.write("Accuracy on Training data : ", training_data_accuracy)
        st.write("Accuracy on Testing data : ", test_data_accuracy)
        if prediction[0] == 0:
            st.write("The machine confirms that person does not have a heart disease.")
        else:
            st.write("The machine confirms that person has a heart disease.")
