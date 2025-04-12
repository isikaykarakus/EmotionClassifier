import streamlit as st
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Load model
model = joblib.load("models/emotion_classifier_pipe_lr.pkl")

# Emoji Dictionary
emotions_emojis = {
    "Anger": "😠",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😄",
    "Sadness": "😢",
    "Surprise": "😲",
    "Ambigious": "🤔"
}

# App title
st.title(" Turkish Emotion Classifier")
st.subheader("Predict the emotion behind your text 🔮")

# User input
sentence = st.text_area("Write a Turkish sentence below:")

# Prediction button
if st.button("Predict Emotion"):
    if sentence:
        # Predict
        prediction = model.predict([sentence])[0]
        prediction_proba = model.predict_proba([sentence])

        # Create DataFrame for probabilities
        proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
        proba_df = proba_df.T.reset_index()
        proba_df.columns = ['Emotion', 'Probability']
        proba_df['Probability'] = proba_df['Probability'].round(3)

        # Find confidence
        confidence = proba_df.loc[proba_df['Emotion'] == prediction, 'Probability'].values[0]

        # Layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(" Original Text")
            st.success(sentence)

            st.subheader(" Prediction")
            emoji = emotions_emojis.get(prediction, "")
            st.success(f"{prediction} {emoji}")

            st.subheader("Confidence Score")
            st.info(f"{confidence * 100:.2f}% confident")

        with col2:
            st.subheader("Prediction Probabilities")

            # Bar Chart
            fig, ax = plt.subplots(figsize=(8,5))
            colors = ['#FF7F50' if emotion == prediction else '#00BFFF' for emotion in proba_df['Emotion']]

            ax.barh(proba_df['Emotion'], proba_df['Probability'], color=colors, edgecolor="white", height=0.5)
            ax.set_xlabel('Probability', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_title('Emotion Prediction Probabilities', fontsize=14, weight='bold')
            ax.invert_yaxis()  # Highest probability on top
            ax.set_facecolor('#0E1117')  # Dark background for plot area
            fig.patch.set_facecolor('#0E1117')  # Dark background for outside
            ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

            # Beautify ticks
            ax.tick_params(colors='white', labelsize=10)

            # Remove top and right border
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('white')
            ax.spines['bottom'].set_color('white')

            st.pyplot(fig)


    else:
        st.warning("Please write a sentence to predict.")
