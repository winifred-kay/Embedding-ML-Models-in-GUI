import streamlit as st
import os
import pickle
import numpy as np
import pandas as pd
import re
from PIL import Image
from sklearn.preprocessing import OneHotEncoder

#Load machine learning component
model = pickle.load(open('C:/Users/PK/Documents/LP4/streamlit app/tmp/Ts_model.sav','rb'))

st.title("Sales Prediction Using Machine Learning")

@st.cache_data
def load_data(relative_path):
   
    merged_data = pd.read_csv(relative_path, index_col= 0)
    merged_data["onpromotion"] = merged_data["onpromotion"].apply(int)
    #merged_data["date"] = pd.to_datetime(merged_data["date"])
    return merged_data
#Extracting date features
@st.cache_data
def getDateFeatures(df, date):
   df["date"] = pd.to_datetime(df["date"])
   df["Year"] = df['date'].dt.year
   df["Month"] = df['date'].dt.month
   df["Week"] = df['date'].dt.week
   df["Day"] = df['date'].dt.day
   df['quarter'] = df['date'].dt.quarter 
   df['week_of_year'] = df['date'].dt.isocalendar().week
   return df

path = r"C:\Users\PK\Documents\LP4\streamlit app\tmp\MergedData.csv"
merged_data = load_data(path)

image = Image.open(r"C:\Users\PK\Documents\LP4\streamlit app\tmp\Predictive.jpg")

header = st.container()
dataset = st.container()
features_and_output = st.container()

#Creating a form to take inputs
form = st.form(key="information", clear_on_submit=True)

with header:
    header.write("App Interface To Predict Retail Sales.")
    header.image(image)

with dataset:
    if dataset.checkbox("Preview the data"):
        dataset.write(merged_data.head())

expected_inputs = ["date",  "family",  "store_cluster",  "events", "onpromotion",  "oil_price", "events"]
categoricals = ["family","events"]

with features_and_output:
    features_and_output.subheader("Input Form")
    features_and_output.write("This form takes all inputs required for prediction")

with form:
    date = st.date_input("Date format e.g. year-month-day")
    Family = st.selectbox("Product family:", options= sorted(list(merged_data["family"].unique())))
    onpromotion = st.number_input("Number of products on promotion:", min_value= merged_data["onpromotion"].min(), value= merged_data["onpromotion"].min())
    store_cluster = st.number_input("Enter store cluster group",min_value=1,max_value=17,step=1)
    oil_price = st.slider("Enter the current oil price",min_value=1.00,max_value=100.00,step=0.1)
    events = st.selectbox("Is the date a holiday or not:", options= sorted(list(merged_data["events"].unique())))

    submitted = form.form_submit_button(label= "Submit")

if submitted:
    with features_and_output:
        # Inputs formatting
        input_dict = {
            "date": [date],
            "family": [Family],
            "cluster": [store_cluster],
            "onpromotion": [onpromotion],
            "oil_price": [oil_price],
            "events": [events]
        }
        
   # Convert inputs into a dataframe
        input_data = pd.DataFrame.from_dict(input_dict)
        input_df = input_data.copy()
   # Converting the date column to the expected datatype
        input_data["date"] = pd.to_datetime(input_data["date"])

    # Getting date features
        df_processed = getDateFeatures(input_data, "date")
        #st.write(df_processed.head())

    # Encoding the categoricals
        oh_encode = OneHotEncoder()     
        encoded = oh_encode.fit_transform(input_data[categoricals])
        encoded = pd.DataFrame(encoded, columns = oh_encode.get_feature_names_out().tolist())
        #df_processed = pd.concat([ df_processed.reset_index(drop=True), encoded], axis=1)
        df_processed = df_processed.join(encoded)
        df_processed.drop(columns=categoricals, inplace=True)
        df_processed.drop(columns="date", inplace=True)
        st.write(df_processed.head())

    # Making the predictions        
        sale_pred = model.predict(df_processed)
        df_processed["sales"] = sale_pred
        input_df["sales"] = sale_pred
        display = sale_pred[0]

    # Adding the predictions to previous predictions
        st.session_state["results"].append(input_df)
        result = pd.concat(st.session_state["results"])


    # Displaying prediction results
    st.success(f"The predicted Sales: {display}")
