import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    * {
        font-weight: 500 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #000080 !important;
        font-weight: 700 !important;
    }
    .stTitle {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    .stSubheader {
        color: #000080 !important;
    }
            
    p {
        color: #000000 !important;
        font-weight: 500 !important;
    }

    span, div, label {
        color: #000080 !important;
        font-weight: 500 !important;
    }
    .stWrite {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Page title and description
st.title("🚢 Titanic Survival Prediction")
st.markdown("---")
st.write("Predict whether a passenger would have survived the Titanic disaster using machine learning.")

# Load the model
try:
    model = joblib.load('Titanic_survival_prediction_model.joblib')
except FileNotFoundError:
    st.error("❌ Model file not found. Please ensure 'Titanic_survival_prediction_model.joblib' is in the same directory.")
    st.stop()

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Passenger Information")
    
    # Passenger Class
    pclass = st.selectbox(
        "Passenger Class",
        options=[1, 2, 3],
        help="1 = Upper class, 2 = Middle class, 3 = Lower class"
    )
    
    # Gender
    sex_input = st.selectbox(
        "Gender",
        options=["Male", "Female"],
        help="Passenger gender"
    )
    sex = 1 if sex_input == "Male" else 0
    
    # Age
    age = st.slider(
        "Age (years)",
        min_value=0,
        max_value=100,
        value=30,
        help="Passenger age"
    )

with col2:
    st.subheader("Additional Details")
    
    # Ticket Fare
    fare = st.number_input(
        "Ticket Fare ($)",
        min_value=0.0,
        value=50.0,
        step=1.0,
        help="Passenger ticket fare amount"
    )
    
    # Family Size
    family_size = st.number_input(
        "Family Size (including self)",
        min_value=1,
        max_value=10,
        value=1,
        help="Number of family members traveling (including the passenger)"
    )

# Prediction section
st.markdown("---")

if st.button("🔮 Predict Survival", use_container_width=True, type="primary"):
    # Prepare input data for the model
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'Fare': [fare],
        'FamilySize': [family_size]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    # Display results
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    col1, col2 = st.columns(2) 
    
    with col1:
        if prediction == 1:
            st.success("✅ **SURVIVED**", icon="✅")
            survival_prob = probability[1] * 100
        else:
            st.error("❌ **DID NOT SURVIVE**", icon="❌")
            survival_prob = probability[0] * 100
    
    with col2:
        st.info(f"🎯 Confidence: **{survival_prob:.1f}%**")
    
    # Display input summary
    st.markdown("---")
    st.subheader("Input Summary")
    
    summary = f"""
    - **Passenger Class**: {pclass}
    - **Gender**: {sex_input}
    - **Age**: {age} years
    - **Ticket Fare**: ${fare:.2f}
    - **Family Size**: {family_size}
    """
    st.info(summary)

# Footer
st.markdown("---")
col_footer1, col_footer2 = st.columns([3, 1])
with col_footer1:
    st.caption("🤖 Model: Decision Tree Classifier | 📚 Data: Titanic Dataset")
with col_footer2:
    st.caption("By **Dinesh**")
