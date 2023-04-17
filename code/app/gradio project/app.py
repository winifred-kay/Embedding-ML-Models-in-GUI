import gradio as gr
import pandas as pd
import numpy
import pickle
import os
import re
from PIL import Image


# Load the saved components
with open("best_knn .pkl", "rb") as f:
    components_clf = pickle.load(f)

# Extract the individual components
num_imputer = components_clf["num_imputer"]
cat_imputer = components_clf["cat_imputer"]
encoder = components_clf["encoder"]
scaler = components_clf["scaler"]
knn_model = components_clf["models"]

#prediction
def process_and_predict(gender,SeniorCitizen,Partner,tenure,Dependents,PhoneService,MultipleLines,
                       InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,
                       Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges):
    
    # Create a dataframe with the input data
    input_df = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges]
        
 })
   # Selecting categorical and numerical columns separately
    cat_columns = [col for col in input_df.columns if input_df[col].dtype == 'object']
    num_columns = [col for col in input_df.columns if input_df[col].dtype != 'object']

    # Apply the imputers on the input data
    input_df_imputed_cat = cat_imputer.transform(input_df[cat_columns])
    input_df_imputed_num = num_imputer.transform(input_df[num_columns])

    # Encode the categorical columns
    input_encoded_df = pd.DataFrame(encoder.transform(input_df_imputed_cat).toarray(),
                                   columns=encoder.get_feature_names(cat_columns))

    # Scale the numerical columns
    input_df_scaled = scaler.transform(input_df_imputed_num)
    input_scaled_df = pd.DataFrame(input_df_scaled , columns = num_columns)
    

    #joining the cat encoded and num scaled
    final_df = pd.concat([input_encoded_df, input_scaled_df], axis=1)

    # Make predictions using the model
    predictions = knn_model.predict(final_df)
     

    return {"predict: CHURN": float(predictions[0]), "predict: Not Churn": 1-float(predictions[0])}

input_interface=[]
with gr.Blocks(css=".gradio-container {background-color: black}") as app:
    
    img = gr.Image("C:\gradio project\Customer-Churn.png").style(height='24')
    Title=gr.Label('CUSTOMER CHURN PREDICTION APP')

    with gr.Row():
        Title
    
    with gr.Row():
        img
    
    

    with gr.Row():  
            
        input_interface.append(gr.components.Dropdown(['male','female'],label='gender')),
        input_interface.append(gr.components.Number(label="Seniorcitizen No=0 and Yes=1")),
        input_interface.append(gr.components.Dropdown(['Yes', 'No'], label='Partner')),
        input_interface.append(gr.components.Dropdown(['No', 'Yes'], label='Dependents')),
        input_interface.append(gr.components.Number(label='Tenure(no.of months)')),
        input_interface.append(gr.components.Dropdown(['No', 'Yes'], label='PhoneService')), 
        input_interface.append(gr.components.Number(label="Enter monthly charges")),
        input_interface.append(gr.components.Number(label="Enter total charges"))  
        
    input_interface.append(gr.components.Dropdown(['No phone service', 'No', 'Yes'], label='MultipleLines')),
    input_interface.append(gr.components.Dropdown(['DSL', 'Fiber optic', 'No'], label='InternetService')),
    input_interface.append(gr.components.Dropdown(['No', 'Yes'], label='OnlineSecurity')),
    input_interface.append(gr.components.Dropdown(['No', 'Yes'], label='OnlineBackup')),
    input_interface.append(gr.components.Dropdown(['No', 'Yes'], label='DeviceProtection')),
    input_interface.append(gr.components.Dropdown(['No', 'Yes'], label='TechSupport')),
    input_interface.append(gr.components.Dropdown(['No', 'Yes'], label='StreamingTV')),
    input_interface.append(gr.components.Dropdown(['No', 'Yes'], label='StreamingMovies')),
    input_interface.append(gr.components.Dropdown(['Month-to-month', 'One year', 'Two year'], label='Contract')),
    input_interface.append(gr.components.Dropdown(['Yes', 'No'], label='PaperlessBilling')),
    input_interface.append(gr.components.Dropdown(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], label='PaymentMethod')),
        

    predict_btn= gr.Button('Predict')
    
    
    # Define the output interfaces
    output_interface = gr.Label(label="Customer churn")
    predict_btn.click(fn=process_and_predict, inputs =input_interface,outputs=output_interface)
  

app.launch()