# streamlit_app
Heart Disease AI Prediction App
This is a Streamlit web app that uses a machine learning model to predict whether a person has a heart disease or not based on patient data.

The model is trained on a dataset of heart disease patients, and uses a logistic regression algorithm to make predictions. The app allows the user to input patient data and displays the prediction along with the model's accuracy on the training and test data.

Getting started
Clone this repository using git clone https://github.com/<username>/heart-disease-prediction-app.git
Install the required libraries by running pip install -r requirements.txt
Run the app using streamlit run app.py
Input format
The app expects the input data to be in the following format:

age ,sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal:
Output format
The app will output the prediction of the model, along with its accuracy on the training and test data. It will also display a message whether person has a heart disease or not.

Licensing
This project is licensed under the MIT License - see the LICENSE.md file for details.

Note
This is a sample app and is not intended for use in a production environment. The model used in this app is not fine-tuned and may not perform well on unseen data. It is recommended to use a more complex model and fine-tune it using a larger dataset for better results.
