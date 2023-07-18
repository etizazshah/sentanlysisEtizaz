import random
import time
import joblib
import pickle
import streamlit as st
from preprocessor import preprocess_text
import nltk

# Load NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Function to load the vectorizer
@st.cache(allow_output_mutation=True)
def load_vectorizer():
    with open("./Models/count_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer

# Function to load the Naive Bayes model
@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load("./Models/naive_model.pkl")
    return model

# Initialize the vectorizer and model
text_vectorizer = load_vectorizer()
naive_model = load_model()

# Building the front end
st.title("Sentiment Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create a random color generator for assistant responses
def get_random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

# Assistant responses for different sentiment classes
positive_responses = [
    "I sense a positive sentiment!",
    "Your message exudes positivity!",
    "A burst of sunshine! Your sentiment is clearly positive!",
    "You're spreading positive vibes! Well done!",
]
negative_responses = [
    "I sense a negative sentiment in your message.",
    "Your message seems negative.",
    "Let's turn things around! Your sentiment appears negative.",
    "Sending virtual hugs! Your message conveys a negative sentiment.",
]
neutral_responses = [
    "Hmm, your message is quite neutral. Keep expressing!",
    "A balanced sentiment! Your message is neither too positive nor too negative.",
    "You're keeping it cool! A neutral sentiment detected.",
    "Your message is like a calm breeze - neutral and composed.",
]

# React to user input
if prompt := st.text_input("Enter your text here"):
    # Display user message in chat container
    st.text_area("User:", prompt, height=100, max_chars=None, key=None)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Analyze sentiment using the model
    new_text = preprocess_text(prompt)
    new_text = [new_text]  # Convert to a list of strings
    new_text_vec = text_vectorizer.transform(new_text)  # Transform directly
    sentiment = naive_model.predict(new_text_vec)

    # Determine sentiment label and corresponding response
    if sentiment[0] == 1:
        response = random.choice(positive_responses)
        color = get_random_color()
    elif sentiment[0] == 0:
        response = random.choice(negative_responses)
        color = get_random_color()
    else:
        response = random.choice(neutral_responses)
        color = get_random_color()

    # Display assistant response in chat container with colored text
    st.text_area("Assistant:", response, height=100, max_chars=None, value=None, color=color)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
