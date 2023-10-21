import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


ssl._create_default_https_context = ssl._create_unverified_context

# Set the NLTK data path and download 'punkt' tokenizer data
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Define intents and responses
intents = [
    {
        "tag": "greeting",
        "patterns": ["hi", "hello", "hey", "howdy"],
        "responses": ["Hi there!", "Hello!", "Hey!", "How can I assist you today?"],
    },
    {
        "tag": "goodbye",
        "patterns": ["goodbye", "bye", "see you later", "farewell"],
        "responses": ["Goodbye!", "See you later!", "Take care!", "Have a great day!"],
    },
    {
        "tag": "thanks",
        "patterns": ["thank you", "thanks", "appreciate it", "you're awesome"],
        "responses": ["You're welcome!", "No problem!", "Anytime!", "Glad I could help!"],
    },
    {
        "tag": "weather",
        "patterns": ["what's the weather like today?", "tell me the weather forecast", "is it going to rain?"],
        "responses": ["I'm sorry, I don't have access to real-time weather information.", "I can't provide weather updates at the moment."],
    },
    {
        "tag": "news",
        "patterns": ["give me the latest news", "what's happening in the world?", "tell me some news headlines"],
        "responses": ["I'm not connected to news sources to provide real-time news updates.", "I can't fetch news for you right now."],
    },
]

vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)
tags = []
patterns = []

for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
clf.fit(x, tags)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]

    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

def main():
    st.title("Rule-Based Chatbot")
    user_input = st.text_input("You: ")

    if st.button("Send"):
        response = chatbot(user_input)
        st.write("Bot:", response)

if __name__ == "__main__":
    main()
