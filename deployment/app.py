import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ============================================================
# Load Model from Hugging Face Model Hub
# ============================================================

model_path = hf_hub_download(
    repo_id="Luthufia/tourism-xgboost-model",
    filename="tourism_xgboost_model.joblib"
)

model = joblib.load(model_path)

# ============================================================
# Streamlit UI
# ============================================================

st.title("Tourism Package Purchase Prediction")

st.write("""
This application predicts whether a customer is likely to purchase
the Wellness Tourism Package based on demographic, travel history,
and customer interaction information.
""")

# ============================================================
# Customer Inputs
# ============================================================

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

typeofcontact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

citytier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

durationofpitch = st.number_input(
    "Duration of Pitch",
    min_value=1,
    max_value=150,
    value=15
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

numberofpersonvisiting = st.number_input(
    "Number Of Persons Visiting",
    min_value=1,
    max_value=10,
    value=2
)

numberoffollowups = st.number_input(
    "Number Of Follow Ups",
    min_value=0,
    max_value=10,
    value=3
)

productpitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
)

preferredpropertystar = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

maritalstatus = st.selectbox(
    "Marital Status",
    ["Married", "Single", "Divorced"]
)

numberoftrips = st.number_input(
    "Number Of Trips",
    min_value=0,
    max_value=25,
    value=3
)

passport = st.selectbox(
    "Passport",
    [0, 1]
)

pitchsatisfactionscore = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)

owncar = st.selectbox(
    "Own Car",
    [0, 1]
)

numberofchildrenvisiting = st.number_input(
    "Number Of Children Visiting",
    min_value=0,
    max_value=5,
    value=1
)

designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

monthlyincome = st.number_input(
    "Monthly Income",
    min_value=1000,
    max_value=100000,
    value=25000
)

# ============================================================
# Prediction
# ============================================================

input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeofcontact,
    "CityTier": citytier,
    "DurationOfPitch": durationofpitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": numberofpersonvisiting,
    "NumberOfFollowups": numberoffollowups,
    "ProductPitched": productpitched,
    "PreferredPropertyStar": preferredpropertystar,
    "MaritalStatus": maritalstatus,
    "NumberOfTrips": numberoftrips,
    "Passport": passport,
    "PitchSatisfactionScore": pitchsatisfactionscore,
    "OwnCar": owncar,
    "NumberOfChildrenVisiting": numberofchildrenvisiting,
    "Designation": designation,
    "MonthlyIncome": monthlyincome
}])

if st.button("Predict"):

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(
            "Customer is likely to purchase the tourism package."
        )
    else:
        st.warning(
            "Customer is unlikely to purchase the tourism package."
        )
