import streamlit as st
import numpy as np
import pickle


# load the saved model
model = pickle.load(open(r'linear_regression_model.pkl','rb'))

# set title of the streamlite app
st.title('Salary Prediction Application')

# add a brief description
st.write('This app predicts the salary based on years of experience using a simple linear regression model')

# input widget for user to enter years of experience
years_experience = st.number_input('enter years of experience:',min_value=0.0,max_value=50.0,value=1.0,step=0.5)


# when button clicked ,make predictions

if st.button('Predicted Salary'):
    
    # make a prediction using the training model
     experience_input=np.array([[years_experience]]) #convert the input to a 2D array for prediction
     prediction = model.predict(experience_input)
     
    # Display results
     st.success(f"The predicted salary for {years_experience} years of experience is: ${prediction[0]:,.2f}")
    
st.write('The model trained using a dataset of salaries and years of experience.Built model by yamini')    