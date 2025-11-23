import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Load the model
model = pickle.load(open("/model/disease_model.pkl", "rb"))

# Load the medicine mapping
with open("medicine_map.json") as f:
    medicine_map = json.load(f)

# Load dataset to get symptom list
df = pd.read_csv("data/Training.csv")
symptoms = df.columns[:-1].tolist()   # all columns except prognosis

st.title("🩺 Disease Prediction & Medicine Recommender")
st.write("Select the symptoms you are experiencing to get a predicted disease and recommended medicine.")

st.header("Select Symptoms")
selected_symptoms = st.multiselect("Choose symptoms:", symptoms)

if st.button("Predict"):
    if not selected_symptoms:
        st.error("Please select at least one symptom!")
    else:
        # Convert symptoms to model input format
        input_data = np.zeros(len(symptoms))
        for s in selected_symptoms:
            input_data[symptoms.index(s)] = 1

        # Predict disease
        prediction = model.predict([input_data])[0]

        # Medicine recommendation
        medicine = medicine_map.get(prediction, "Consult a doctor for appropriate treatment.")

        st.success(f"### 🧬 Predicted Disease: **{prediction}**")
        st.info(f"### 💊 Recommended Medicine: **{medicine}**")







