# Turkish Emotion Classifier

A complete **Natural Language Processing (NLP)** project that classifies Turkish text into emotions such as **happy**, **sadness**, **anger**, **fear**, **disgust**, **surprise**, and **ambiguous**.  
The model is trained on a labeled dataset and deployed with a lightweight **Streamlit** web application for real-time prediction.

---

## Project Structure

```
turkish_emotion_classifier/
├── TurkishEmotionClassifier_SingleLabel.ipynb
├── models/
│   └── emotion_classifier_pipe_lr.pkl
├── images/
│   └── DemoPicture.png
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── TREMODATA.xml
│   └── paper_related_to_TREMO_dataset.pdf
```

---

## Pipeline

1. **Data Loading**  
   Parse and load XML data from the **TREMO** dataset.

2. **Data Preprocessing**  
   - Lowercasing text  
   - Removing punctuation (using `nfx` + manual cleaning)  
   - Removing stopwords  
   - Removing numbers and extra whitespace

3. **Feature Extraction**  
   Convert cleaned text into numerical features using **CountVectorizer**.

4. **Model Training**  
   Train a **Logistic Regression** classifier to predict emotions.

5. **Model Evaluation**  
   Evaluate using Confusion Matrix, Accuracy, Precision, Recall, and F1-score.  
   Achieved **~82% accuracy** with strong performance across major emotion classes.

6. **Web App Deployment**  
   Built a **Streamlit** app where users can input Turkish text and receive real-time emotion predictions with probability scores.

---

## Libraries

- **Python** >= 3.9
- **Scikit-learn** 1.2.2
- **Streamlit** 1.22.0
- **NLTK** 3.8.1
- **Pandas** 1.5.3
- **Matplotlib** 3.7.1
- **Seaborn** 0.12.2
- **Neattext** 0.1.3
- **Joblib** 1.2.0

---

## Results

- **Accuracy**: ~82% on the test set
- **Macro F1-Score**: ~76%
- **Observations**:
  - Strong performance on main emotions such as **Happy**, **Sadness**, **Fear**, and **Disgust**.
  - **Ambiguous** class remains challenging.
  - Some sentences express multiple emotions; however, the model was trained for **single-label classification**.

---

##  Live Demo

![App Screenshot](https://github.com/isikaykarakus/EmotionClassifier/blob/main/images/DemoPicture.png)

 **[Click Here](https://turkishemotionclassifier-isikaykarakus.streamlit.app/)**

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/isikaykarakus/EmotionClassifier.git
   cd EmotionClassifier
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

The app will open automatically in your browser.

---

## Future Improvements

- Implement **multi-label classification** to handle overlapping emotions.
- Fine-tune with advanced models like **BERTurk** or **XLM-Roberta**.
- Expand the dataset with more nuanced emotion labels.
- Deploy the app online using **Streamlit Cloud** for public access.

---

## Acknowledgments

- **[TREMO Dataset](https://www.kaggle.com/datasets/mansuralp/tremo)** for providing Turkish emotional text data.

---

## Author

**Işıkay Karakuş**  
- [GitHub](https://github.com/isikaykarakus)  
- [LinkedIn](https://www.linkedin.com/in/isikaykarakus/)
