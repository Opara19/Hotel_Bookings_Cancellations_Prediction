import streamlit as st
import pandas as pd
import pymongo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import folium 
from streamlit_folium import folium_static
import json
from matplotlib import cm, colors
from streamlit_option_menu import option_menu
from PIL import Image
import altair as alt
from datetime import datetime
from pymongo import MongoClient
import time
import pickle

if 'data' not in st.session_state:
    st.session_state['data'] = None

mongo_url = "mongodb+srv://opara_862:Mongodb8@cluster0.9mmkw4y.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(mongo_url)
db = client['Hotel_bookings']
collection = db['Customer_data']

st.set_page_config(page_title="VINIRA Reservations Cancellation Predictions", 
                   layout="wide",
                   initial_sidebar_state='expanded')

st.markdown(
    """
    <div style="text-align: center; margin: 20px; background-color:rgb(34,193,195); padding: 20px;width:100%">
        <h1>VINIRA Reservations Cancellation Predictions</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Load data from MongoDB
@st.cache_data
def load_data():
    data = list(collection.find())
    df = pd.DataFrame(data)
    df=df.drop('_id',axis=1)
    return df

def prepare_data_for_prediction(df):
    processed_df = df.copy()
    categorical_mappings = {
        'type_of_meal_plan':['Meal Plan 2','Meal Plan 3','Not Selected'],
        'room_type_reserved':['Room_Type 2','Room_Type 3','Room_Type 4','Room_Type 5','Room_Type 6','Room_Type 7'],
        'market_segment_type': ['Online', 'Offline','Complementary','Corporate'],
        'arrival_month':['January','February','March','May','June','July','August','September',
        'October','November','December'],
                
    }
    for column, values in categorical_mappings.items():
        for value in values:
            column_name = f"{column}_{value}"
            processed_df[column_name] = (processed_df[column] == value).astype(int)
       
        processed_df = processed_df.drop(column, axis=1)

    with open('feature_names.pkl', 'rb') as f:
        expected_feature_names = pickle.load(f)

    processed_df = processed_df.reindex(columns=expected_feature_names, fill_value=0)

    processed_df
    X = processed_df.copy()

    
    return X

def make_predictions(data):
    """Load the model and make predictions"""
    try:
        model = pickle.load(open('finalized_model.sav', 'rb'))
    except FileNotFoundError:
        st.error("Model file 'finalized_model.sav' not found!")
        return None

    X = prepare_data_for_prediction(data)

    probabilities = model.predict_proba(X)
    
    # Return cancellation probabilities (probability of class 1)
    return probabilities[:, 1]
# def load_model():
#     with open("finalized_model.sav", "rb") as file:
#         model = pickle.load(file)
#     return model

# model = load_model()
coln1, coln2 = st.columns(2)
# df=None




with coln1:
    if st.button("Load the Customer Data"):
        df = load_data()
        st.session_state['data'] = df
        
        # Display the data
        st.dataframe(df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
with coln2:
    if st.button("Predict Cancellation Probability"):
        if st.session_state['data'] is None:
            st.error("Please load the customer data first!")
        else:
            # Make predictions
            predictions = make_predictions(st.session_state['data'])
            
            if predictions is not None:
                # Add predictions to the dataframe
                result_df = st.session_state['data'].copy()
                result_df['Cancellation_Probability'] = predictions
                
                # Display results
                st.write("### Prediction Results")
                st.dataframe(
                    result_df[['Cancellation_Probability']],
                    use_container_width=True
                )
                
                # Display summary statistics
                # st.write("### Summary Statistics")
                # st.write(f"Average Cancellation Probability: {predictions.mean():.2%}")
                # st.write(f"Number of High-Risk Bookings (>50% probability): {(predictions > 0.5).sum()}")
                
                # # Optional: Create a histogram of probabilities
                # fig, ax = plt.subplots()
                # ax.hist(predictions, bins=20)
                # ax.set_xlabel('Cancellation Probability')
                # ax.set_ylabel('Count')
                # ax.set_title('Distribution of Cancellation Probabilities')
                # st.pyplot(fig)
                
        st.markdown("</div>", unsafe_allow_html=True)
# Add some custom CSS for styling (optional)
st.markdown(
    """
    <style>
    .stButton > button {
        width:100%;
        background-color: #090979;
        color: white;
        margin: 20px;
        font-size: 20px;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        height:50px;
    }
    .stButton > button:hover {
        background-color: #eeaeca; 
        color:black;
    }
    </style>
    """,
    unsafe_allow_html=True
)




# # Add a title and image inside a div container
# st.markdown(
#     """
#     <div style="text-align: center; margin: 20px; background-color: brown; padding: 20px;">
#         <h1>VINIRA Reservations Cancellation Predictions</h1>
#     </div>
#     """,
#     unsafe_allow_html=True
# )
# # Create a button to load the customer data
# if st.button("Load the Customer Data"):
#     df = load_data()
#     # Display the DataFrame
#     st.dataframe(df)

# # Add some custom CSS for styling (optional)
# st.markdown(
#     """
#     <style>
#     .stButton > button {
#         background-color: brown; /* Green */
#         color: black;
#         margin: 20px;
#         font-size: 20px;
#         padding: 10px 20px;
#         border: none;
#         border-radius: 5px;
#         cursor: pointer;
#     }
#     .stButton > button:hover {
#         background-color: #45a049;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# texts = [
#     "Welcome to VINIRA Reservations Cancellation Predictions!",
#     "Stay tuned for the latest predictions.",
#     "Our goal is to provide accurate cancellation forecasts.",
#     "Thank you for visiting our application!"
# ]

# # Create a placeholder for the carousel
# carousel_placeholder = st.empty()

# # Create a function to run the carousel
# def text_carousel(texts, interval=1):
#     while True:
#         for text in texts:
#             # Update the placeholder with the current text
#             carousel_placeholder.markdown(
#                 f"<div style='text-align: center; font-size: 24px; padding: 20px; color: white;'>{text}</div>",
#                 unsafe_allow_html=True
#             )
#             # Wait for the specified interval before showing the next text
#             time.sleep(interval)

# # Start the carousel
# if st.button("Start Carousel"):
#     text_carousel(texts)