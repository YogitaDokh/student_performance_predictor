import streamlit as st
import pickle
import numpy as np

# Set up the page title and description
st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("🎓 Student Performance Predictor")
st.write("Enter the details below to estimate the student's score based on the trained Linear Regression model.")

# Load the trained model safely
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
    
    # Create a nice layout with columns for inputs
    st.subheader("📝 Input Features")
    col1, col2 = st.columns(2)
    
    with col1:
        hours_studied = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
        previous_scores = st.number_input("Previous Scores", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
        
    with col2:
        sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        papers_practiced = st.number_input("Sample Question Papers Practiced", min_value=0, max_value=20, value=2, step=1)

    # Separation line
    st.markdown("---")

    # Prediction button
    if st.button("Predict Score", type="primary"):
        # Arrange inputs in the exact order the model expects
        input_data = np.array([[hours_studied, previous_scores, sleep_hours, papers_practiced]])
        
        # Make the prediction
        prediction = model.predict(input_data)[0]
        
        # Ensure the score stays within a realistic logical boundary (e.g., 0 to 100)
        final_score = max(0.0, min(100.0, float(prediction)))
        
        # Display the result
        st.success(f"🎯 **Predicted Performance Score:** {final_score:.2f}")

except FileNotFoundError:
    st.error("⚠️ **Error:** `model.pkl` not found. Please ensure the model file is in the same directory as this script.")
except Exception as e:
    st.error(f"❌ An error occurred while loading the model or making a prediction: {e}")
