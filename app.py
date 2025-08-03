## import necessary library.
import pandas as pd
import numpy as np
import streamlit as st
import pickle
from PIL import Image

## load saved model.
with open("customer_churn_model.pkl",'rb') as f:
    load_model = pickle.load(f)
print(load_model)


model = load_model['model']
print(load_model)


features_names = load_model['features_names']
# print(features_names)


## Load saved Label Encoder
with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)
print(encoders)


# Set Title and Header
img = Image.open('BSNL transparent ora.png')
st.image(img, width=150) # company logo.
st.title('Bharat Sanchar Nigam Limited') # without any intention i am using BSNL company name.
st.subheader("Welcome to BSNL India.")
st.header("Customer Churn Prediction.")
st.subheader('Enter customer details')


## insert form to get user data or customer data.
with st.form('fill details'):
    gender = st.radio("Select customer gender:",["Male","Female"])
    senior_citizen = st.radio("Is customer senior citizen, 'Yes' means 1 and 'No' means 0",["Yes","No"])
    senior_cit = senior_citizen # we will use it, it will help to show user filled data.
    dependent = st.radio("Has any any dependent on customer:", ["Yes","No"])

    tenure = st.number_input("Enter customer tenure: ",min_value=0) # age or time never will be negative. handle it use minimum value is 0.
    phone_service = st.radio("Phone service: ",["Yes","No"])
    MultiPleLines = st.radio("Multiple Lines:", ['No phone service', 'No', 'Yes'])
    InternetServices = st.radio("Internet Services:", ['DSL', 'Fiber optic', 'No'])

    TechSupport = st.radio("Tech support:", ['No', 'Yes', 'No internet service'])
    Onlinebackup = st.radio("Have Customer online Backup: ", ['Yes', 'No', 'No internet service'])
    streaming_tv = st.radio("Customer is streaming Tv:",["Yes","No"])
    Contract = st.radio("Contract time duration:", ['Month-to-month', 'One year', 'Two year'])

    PaymentMethod = st.radio("select Payment Method:",['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
                                                       'Credit card (automatic)'])
    monthly_charges = st.number_input("Enter customer Monthly charges: ",min_value=0)
    total_charges = st.number_input("Enter customer Total Charges: ", min_value=0)
    submitted = st.form_submit_button("Submit")


    ## convert yes and no into 1 and 0. Where for yes 1 and for no 0.
    if senior_citizen == "Yes":
        senior_citizen = 1

    elif senior_citizen == "No":
        senior_citizen = 0

    else:
        print("invalid input.")


## We need to create dictionary for that we will declare key and value.
keys = ['gender','SeniorCitizen','Dependents','tenure','PhoneService','MultipleLines','InternetService','TechSupport',
          'OnlineBackup','StreamingTV','Contract','PaymentMethod','MonthlyCharges','TotalCharges']
values = [gender,senior_citizen,dependent,tenure, phone_service,MultiPleLines,InternetServices,TechSupport,
                          Onlinebackup,streaming_tv,Contract,PaymentMethod,monthly_charges,total_charges]


## Create a dictionary of Key and values pair.
input_data = dict(zip(keys,values))
# st.write(input_data)


## Now make Data frame of this dictionary.
input_data = pd.DataFrame([input_data])
if submitted: # submit
    values_1 = [gender,senior_citizen,dependent,tenure, phone_service,MultiPleLines,InternetServices,TechSupport,
                          Onlinebackup,streaming_tv,Contract,PaymentMethod,monthly_charges,total_charges]
    input_data_1 = dict(zip(keys, values_1))
    input_data_1 = pd.DataFrame([input_data_1])
    st.write(input_data_1)
    # print(input_data.columns)
    # print(input_data)


    # input_data_df = st.dataframe(input_data)
    # st.write(input_data_df)


    for column, encoder in encoders.items():
        if column in input_data.columns:
            new_classes = set(input_data[column]) - set(encoder.classes_)
            if new_classes:
                encoder.classes_ = np.append(encoder.classes_, list(new_classes))
            input_data[column] = encoder.transform(input_data[column])


    # Define a function for prediction. Which provide meaningful
    def prediction(prediction):
        if prediction == 0:
            message = "The customer less likely or not churn."
        elif prediction == 1:
            message = "This customer most likely to churn."
        else:
            message = "Please enter valid input."
        return message


    # st.write(dir(model))


    ## apply label encoding on input data frame
    data = input_data[load_model['features_names']]
    # st.write(data)


    # st.write(load_model['features_names'])


    ## Show model prediction
    pred = prediction(model.predict(data))
    st.write(pred)

