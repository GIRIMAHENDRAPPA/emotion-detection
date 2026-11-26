# Emotion Detection

A deep learning app that detects emotion from text using an LSTM model.

## Emotions Detected
Joy, Sadness, Anger, Fear, Love, Surprise

## How It Works
1. User types a sentence in the Streamlit app
2. Text is cleaned and tokenized using `tokenizer.pkl`
3. LSTM model `emotion_detection_lstm.keras` predicts the emotion
4. `label_encoder.pkl` converts the prediction back to an emotion name
5. App shows the predicted emotion with confidence

## Tech Stack
- Python, TensorFlow / Keras, LSTM
- Streamlit for web app
- NLTK for text preprocessing

## How To Run
```bash
pip install -r requirements.txt
streamlit run app.py
