import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Emotion Detection", page_icon="😊")
st.title("😊 Emotion Detection from Text")
st.write("Type a sentence and this model will predict the emotion using your LSTM model.")

@st.cache_resource
def load_all():
    model = load_model("emotion_detection_lstm.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, tokenizer, le

model, tokenizer, le = load_all()

text = st.text_area("Enter your sentence here:", "I am feeling very happy today!")

if st.button("Predict Emotion"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
        pred = model.predict(padded)
        idx = np.argmax(pred[0])
        emotion = le.inverse_transform([idx])[0]
        conf = float(np.max(pred[0]) * 100)

        st.success(f"Predicted Emotion: **{emotion}**")
        st.write(f"Confidence: {conf:.2f}%")
        st.bar_chart(pred[0])
