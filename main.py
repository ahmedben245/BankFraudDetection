import streamlit as st
from pycaret.anomaly import load_model, predict_model
import pandas as pd
from preprocessing_function import preprocessing_function

# Load the dataset
original_data = pd.read_csv('data/bank_transactions_data_2.csv')

# Set the title of the Streamlit app
st.title("Bank Fraud Detection")

# Define the model path
model_path = 'model/mcd_model'
# Load your trained model
anomaly_model = load_model(model_path)

# File uploader widget
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    df_test = pd.read_csv(uploaded_file)
    
    # Display the DataFrame
    st.write("DataFrame Preview:")
    st.dataframe(df_test)
    
    # Apply pre-processing function
    df_test = preprocessing_function(df_test, original_data)

    # Apply predictions
    try:
        df_predict = predict_model(anomaly_model, data=df_test)  # Predict Anomalies

        # Display the results
        st.write("Predictions:")
        columns_to_display = ['TransactionAmount', 'TransactionDuration', 'AccountBalance', 'CustomerAge', 'Anomaly', 'Anomaly_Score']
        st.dataframe(df_predict[columns_to_display])

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
