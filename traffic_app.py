# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gzip

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('train_df.csv')

# reg_pickle = open('traffic_mapie.pickle', 'rb') 
# reg = pickle.load(reg_pickle) 
# reg_pickle.close()

# Path to your compressed file
file_path = "traffic_mapie_compressed.pickle.gz"

# Load the compressed file
with gzip.open(file_path, "rb") as f:
    reg = pickle.load(f)

# used chat gpt and online forums to customize and center this
st.markdown(
    """
    <h1 style="
        text-align: center; 
        background: linear-gradient(to right, red, orange, yellow, green); 
        -webkit-background-clip: text; 
        color: transparent;
    ">
        Traffic Volume Predictor
    </h1>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <h3 style="
        text-align: center; 
        font-size: 30px;
        color: white;
    ">
        Utilize our advanced Machine Learning application to predict traffic volume.
    </h3>
    """,
    unsafe_allow_html=True
)


st.image('traffic_image.gif')


st.sidebar.image('traffic_sidebar.jpg', caption='Traffic Volume Predictor')
st.sidebar.subheader('Input Features')
st.sidebar.write('You can either upload your data file or manually enter input features.')

# option 1 expander
with st.sidebar.expander(label='Option 1: Upload a CSV file'):
  file = st.file_uploader("Upload a CSV file containing traffic details.", type='csv')

  # sample dataframe
  st.write("### Sample Data Format For Upload")
  st.dataframe(df.head(5))
  st.warning('**Ensure your dataframe has the same columns and data types as shown above**', icon='⚠️')


# option 2 expander
with st.sidebar.expander(label='Option 2: Fill Out Form'):

  st.write('Enter the traffic details using the form below')

  # form
  with st.form('user_inputs'):
    holiday = st.selectbox('Choose whether today is a designated holiday or not.', options=df['holiday'].unique())
    temp = st.number_input('Average temperature in kelvin', min_value= int(df['temp'].min()), max_value= int(df['temp'].max()), step=1)
    rain = st.number_input('Amount in mm of rain that occured in the hour', min_value= df['rain_1h'].min(), max_value= df['rain_1h'].max(), step=.1)
    snow = st.number_input('Amount in mm of snow that occured in the hour', min_value= df['snow_1h'].min(), max_value= df['snow_1h'].max(), step=.1)
    cloud = st.number_input('Percentage of cloud cover', min_value= df['clouds_all'].min(), max_value= df['clouds_all'].max(), step=1)
    weather = st.selectbox('Choose the current weather', options=df['weather_main'].unique())
    month = st.selectbox('Choose month', options=df['month'].unique())
    weekday = st.selectbox('Choose weekday', options=df['weekday'].unique())
    hour = st.selectbox('Choose hour', options=df['hour'].unique())

    # submit button
    submit_button = st.form_submit_button("Submit form data", disabled=(file is not None))


if file is None and submit_button == False:
    st.info('**Please choose a data input method to proceed.**', icon='ℹ️')

if file is None and submit_button == True:
    st.success('**Form data submitted successfuly.**', icon='✅')

if file is not None and submit_button == False:
    st.success('CSV file uploaded successfuly.', icon='✅')

alpha = st.slider('Select an alpha value for prediction intervals', min_value=0.01, max_value=0.5, step=.01)

if file is None:
    # Encode the inputs for model prediction
    encode_df = df.copy()
    encode_df = encode_df.drop(columns=['traffic_volume'])

    # Combine the list of user data as a row to default_df
    encode_df.loc[len(encode_df)] = [holiday, temp, rain, snow, cloud, weather, month, weekday, hour]

    # Create dummies for encode_df
    cat_var = ['holiday', 'weather_main', 'month', 'weekday', 'hour']
    encode_dummy_df = pd.get_dummies(encode_df, columns=cat_var, drop_first=True)

    # Extract encoded user data
    user_encoded_df = encode_dummy_df.tail(1)

    y_test_pred, y_test = reg.predict(user_encoded_df, alpha=alpha)

    pred_value = y_test_pred[0]
    lower_limit = y_test[:, 0]
    upper_limit = y_test[:, 1]

    # Ensure limits are within [0, 1]
    lower_limit = max(0, lower_limit[0][0])
    upper_limit = min(1, upper_limit[0][0])

    # Show the prediction on the app
    st.write("## Predicting Traffic Volume...")

    # Display results using metric card
    st.metric(label = "Predicted Traffic Volume", value = f"{pred_value:.0f}")
    st.markdown(f"**Confidence Interval** ({(1-alpha)*100:.1f}%): [{lower_limit:,.2f}, {upper_limit:,.2f}]")

if file is not None and submit_button== False:
    # If a file is uploaded, read it into a DataFrame
    uploaded_file = pd.read_csv(file)
    og_df_copy = df.copy() 

    # Remove the target column from the original data for model input
    og_df_copy_input = og_df_copy.drop(columns=['traffic_volume'])
    
    # Ensure the uploaded data has the same columns as the original data (feature columns only)
    uploaded_file = uploaded_file[og_df_copy_input.columns]
    
    # Concatenate the two DataFrames along rows
    combined_df = pd.concat([og_df_copy_input, uploaded_file], axis=0)
    
    # Number of rows in the original DataFrame (to split later)
    original_rows = og_df_copy_input.shape[0]
    
    # One-hot encode the combined DataFrame
    cat_var = ['holiday', 'weather_main', 'month', 'weekday', 'hour']
    combined_df_encoded = pd.get_dummies(combined_df, columns=cat_var, drop_first=True)
    
    # Split the combined encoded DataFrame into original and user parts
    user_df_encoded = combined_df_encoded[original_rows:]

    y_test_pred, y_test = reg.predict(user_df_encoded, alpha=alpha)

    pred_value = y_test_pred[0]
    lower_limit = y_test[:, 0]
    upper_limit = y_test[:, 1]

    uploaded_file['Predicted Volume'] = y_test_pred.round(1)
    uploaded_file['Lower Limit'] = lower_limit.round(1)
    uploaded_file['Upper Limit'] = upper_limit.round(1)

    # Display the DataFrame with prediction results on the main page
    st.subheader("Prediction Results with 90% Confidence Interval")
    st.dataframe(uploaded_file, width=1000)

# Showing additional items in tabs
st.subheader("Prediction Performance")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Histogram of Residuals", "Predicted vs Actual", "Coverage Plot"])

with tab1:
    st.write("### Feature Importance")
    st.image('traffic_feature_imp.svg')
    st.caption("Features used in this prediction are ranked by relative importance.")

with tab2:
    st.write("### Histogram of Residuals")
    st.image('traffic_dist.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")

with tab3:
    st.write("### Predicted vs Actual")
    st.image('traffic_predvsactual.svg')
    st.caption("Visual comparison of predicted and actual values.")

with tab4:
    st.write("### Coverage Plot")
    st.image('traffic_coverage.svg')
    st.caption("Range of prediction with confidence intervals.")