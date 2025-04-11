import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load the MLP model
model_path = "models/MLPmodel.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Load the dataset
csv_path = "data/processed/Processed_Admission_Dataset.csv"
df = pd.read_csv(csv_path)

# Custom title section with blue background
st.markdown(
    """
    <div style="background-color:#1f77b4;padding:20px;border-radius:10px">
    <h1 style="color:white;text-align:center;">Admission Predictor</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="font-size:16px;color:#333;text-align:center;margin-top:10px;">
    This application predicts your chances of admission based on your scores and other factors.
    </p>
    """,
    unsafe_allow_html=True
)
st.markdown("""
    <style>
        /* Custom Background */
        body {
            background: linear-gradient(135deg, #f0f4f8, #c2c7d0);
            font-family: 'Arial', sans-serif;
        }
        
        .title {
            color: #1f77b4;
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            margin-top: 20px;
        }

        .subheader {
            text-align: center;
            color: #333;
            font-size: 1.5em;
        }

        .input-container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }

        .input-label {
            font-weight: bold;
            color: #1f77b4;
        }

        .stButton button {
            background-color: #1f77b4;
            color: white;
            padding: 12px 24px;
            font-size: 1.1em;
            border-radius: 5px;
            border: none;
        }

        .stButton button:hover {
            background-color: #155a8a;
        }

        .result-text {
            font-size: 1.2em;
            color: #28a745;
            text-align: center;
            margin-top: 20px;
        }

        .image-container {
            text-align: center;
            margin-top: 30px;
        }

        .image-container img {
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        .stTextInput, .stNumberInput, .stSelectbox, .stTextArea {
        background-color: #f5f5f5;
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 10px;
        font-size: 14px;
        color: #333;
    }
    .stTextInput:hover, .stNumberInput:hover, .stSelectbox:hover, .stTextArea:hover {
        border-color: #1f77b4;
    }
    .stSelectbox select, .stNumberInput input {
        background-color: #ffffff;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)
# Form for entering features
with st.form("prediction_form"):
    st.subheader("Enter Your Informations:")
    
    # Custom styling for input fields
    GRE_Score = st.number_input("GRE Score", min_value=0, max_value=340, step=1)
    TOEFL_Score = st.number_input("TOEFL Score", min_value=0, max_value=120, step=1)
    SOP = st.slider("Statement of Purpose (SOP)", min_value=1.0, max_value=5.0, step=0.1)
    LOR = st.slider("Letter of Recommendation (LOR)", min_value=1.0, max_value=5.0, step=0.1)
    CGPA = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.01)
    
    University_Rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])
    Research = st.radio("Do you have research experience?", ["No", "Yes"])
    
    submitted = st.form_submit_button("Predict Admission")

# When form is submitted, show the prediction result
if submitted:
    # Convert values into model format
    University_Rating_1 = 1 if University_Rating == 1 else 0
    University_Rating_2 = 1 if University_Rating == 2 else 0
    University_Rating_3 = 1 if University_Rating == 3 else 0
    University_Rating_4 = 1 if University_Rating == 4 else 0
    University_Rating_5 = 1 if University_Rating == 5 else 0
    Research_0 = 1 if Research == "No" else 0
    Research_1 = 1 if Research == "Yes" else 0
    
    input_data = [[GRE_Score, TOEFL_Score, SOP, LOR, CGPA,
                   University_Rating_1, University_Rating_2, University_Rating_3,
                   University_Rating_4, University_Rating_5, Research_0, Research_1]]
    
    prediction = model.predict(input_data)[0]
    
    st.subheader("Prediction Result:")

    if prediction==1:
        st.success(f"You are admissible")
    elif prediction==0:
        st.error(f"Sorry You are not admissible")
    else:
        st.error(f"Sorry You are not admissible")
    
    # Display images with some style
    st.image("Loss_Curve.png", caption="Loss Curve")
    st.image("gpA.png", caption="CGPA vs Admission")


# Provide additional information about the model
st.markdown(
    """
    <div style="background-color:#f0f2f6;padding:10px 20px;border-radius:10px;margin-top:30px">
    <h4 style="color:#1f77b4;">How does this work?</h4>
    <p style="color:#333;">
    We used a machine learning (Neural Network) model to predict your admissibility. The features used in this prediction are ranked by their relative importance below:
    </p>
    </div>
    """,
    unsafe_allow_html=True
)

