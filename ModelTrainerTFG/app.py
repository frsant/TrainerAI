import streamlit as st
import pickle

# Load the trained model
with open("model_recommendation_model.pkl", "rb") as file:
    model = pickle.load(file)

def recommend_model(description):
    return model.predict([description])[0]

st.title("Model Recommendation System")

user_input = st.text_area("Describe what you want to predict and we will give you a suggestion on what model to choose")

if st.button("Get Recommendation"):
    if user_input:
        recommendation = recommend_model(user_input)
        st.write(f"Recommended Model: {recommendation}")
    else:
        st.write("Please enter a description to get a recommendation.")
