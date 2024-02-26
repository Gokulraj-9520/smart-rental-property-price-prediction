import streamlit as st
from pycaret.regression import *
import pandas as pd
import numpy as np
import warnings
import pickle
warnings.filterwarnings("ignore")

pd.options.display.max_columns=None
pd.options.display.max_rows=None
pd.options.display.max_colwidth=None

st.title("Smart Rental Property Price Prediction")
st.markdown(
    """<div style="text-align: right; margin-right: 50px;">
        <h3 style="color: #1E90FF;">- Created by Gokulraj Pandiyarajan</h3>
    </div>""",
    unsafe_allow_html=True
)
st.write()

st.markdown(
    """
    <style>
    .stButton {
        color: green; /* Set the text color */
        padding: 10px 20px; /* Adjust padding to increase button size */
        font-size: 18px; /* Set the font size */    
        border-radius: 10px; /* Add rounded corners */
    }
    </style>
    """,
    unsafe_allow_html=True
)

types=st.selectbox('Select The House Type',('BHK2', 'BHK3', 'BHK1', 'RK1', 'BHK4', 'BHK4PLUS'))
lease_type=st.selectbox("Select The Lease Type of House",("ANYONE", 'FAMILY', 'BACHELOR', 'COMPANY'))
furnishing=st.selectbox("Select The Furnishing of House",('SEMI_FURNISHED', 'FULLY_FURNISHED', 'NOT_FURNISHED'))
facing=st.selectbox("Select The House Door Facing side",('NE', 'E', 'S', 'N', 'SE', 'W', 'NW', 'SW'))
parking=st.selectbox("Select Parking Type",('BOTH', 'TWO_WHEELER', 'NONE', 'FOUR_WHEELER'))
water_supply=st.selectbox("Select Type of Water Supply",('CORPORATION', 'CORP_BORE', 'BOREWELL'))
building_type=st.selectbox("Select The Building Type",('AP', 'IH', 'IF', 'GC'))
activation_date=st.date_input("Select The House Activation Date")
latitude=st.number_input('Latitude',min_value=12.9000000,max_value=13.0000000,step=0.00000001,format="%.8f")
longitude=st.number_input('Longitude',min_value=77.50000000,max_value=77.76607800,step=0.00000001,format='%.8f')
negotiable=st.checkbox("Negotiable",0)
property_size=st.number_input("Property Size",min_value=250.000000,max_value=50000.000000,step=0.000001,format='%.6f')
bathroom=st.number_input("Number of Bathroom",min_value=1,max_value=4,step=1)
cup_board=st.number_input("Number of Cup Boards",min_value=0,max_value=8,step=1)
property_age=st.number_input("Age of Property",min_value=0, max_value=25,step=1)
floor=st.number_input("Number of Floors",min_value=0,max_value=11,step=1)
total_floor=st.number_input("Total Number of Floors",min_value=0, max_value=17,step=1)
balconies=st.number_input("Number of Balconies",min_value=0,max_value=4,step=1)
LIFT=st.checkbox("LIFT",False)
GYM=st.checkbox("GYM",False)
INTERNET=st.checkbox("INTERNET",False)
AC=st.checkbox("AC",False)
CLUB=st.checkbox("CLUB",False)
INTERCOM=st.checkbox("INTERCOM",False)
POOL=st.checkbox("POOL",False)
CPA=st.checkbox("CPA",False)
FS=st.checkbox("FS",False)
SERVANT=st.checkbox("SERVANT",False)
SECURITY=st.checkbox("SECURITY",False)
SC=st.checkbox("SC",False)
GP=st.checkbox("GP",False)
PARK=st.checkbox("PARK",False)
RWH=st.checkbox("RWH",False)
STP=st.checkbox("STP",False)
HK=st.checkbox("HK",False)
PB=st.checkbox("PB",False)
VP=st.checkbox("VP",False)
submit=st.button("submit")
if submit:
    
    new_data={'type':type,
            'activation_date':activation_date,
            'latitude':latitude,
            'longitude':longitude,
            'lease_type':lease_type,
            'negotiable':negotiable,
            'furnishing':furnishing,
            'parking':parking,
            'property_size':property_size,
            'property_age':property_age,
            'bathroom':bathroom,
            'facing':facing,
            'cup_board':cup_board,
            'floor':floor,
            'total_floor':total_floor,
            'water_supply':water_supply,
            'building_type':building_type,
            'balconies':balconies,
            'LIFT':LIFT,
            'GYM':GYM,
            'INTERNET':INTERNET,
            'AC':AC,
            'CLUB':CLUB,
            'INTERCOM':INTERCOM,
            'POOL':POOL,
            'CPA':CPA,
            'FS':FS,
            'SERVANT':SERVANT,
            'SECURITY':SECURITY,
            'SC':SC,
            'GP':GP,
            'PARK':PARK,
            'RWH':RWH,
            'STP':STP,
            'HK':HK,
            'PB':PB,
            'VP':VP
            }
    
    new_data=pd.DataFrame([new_data])
    new_data['activation_date']=pd.to_datetime(new_data['activation_date'])
    predict_lgbm_model = load_model('lightgbm')  # Replace 'your_model_name' with the actual name of your trained model
    prediction = predict_model(predict_lgbm_model, data=new_data)
    st.write(f"Resale Price of Your House \"{round(prediction.prediction_label[0],4)}\"")
