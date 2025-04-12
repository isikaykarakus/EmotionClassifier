
# Turkish Emotion Classifier

A simple Natural Language Processing (NLP) project that classifies Turkish text into emotions like **happy**, **sad**, **angry**, and **neutral**.  
The model is trained on a labeled dataset and deployed with a lightweight **Streamlit** web application for real-time prediction.

---

## Project Structure

```
/turkish-text-classifier
  ├── TurkishEmotionClassifier.ipynb
  ├── data/
  │   ├──TREMODATA.xml
  │   └── paper_related_to_TREMO_dateset.pdf
  ├── app/
  │   └── app.py
  ├── model.pkl
  ├── vectorizer.pkl
  ├── README.md
  ├── requirements.txt
```

---

## How It Works

1. **Data Preprocessing**  
   - Lowercasing text
   - Removing punctuation
   - Tokenizing words

2. **Feature Extraction**  
   - Convert text into numerical features using **CountVectorizer**.

3. **Model Training**  
   - Train a **Logistic Regression** classifier on the extracted features.

4. **Model Evaluation**  
   - Evaluate using accuracy, precision, recall, and F1-score.

5. **Web App Deployment**  
   - Create an interactive user interface with **Streamlit** where users can input Turkish sentences and receive emotion predictions.

---

## Technologies Used

- Python
- Scikit-learn
- NLTK
- Streamlit
- Pandas
- Machine Learning (Logistic Regression)
- Natural Language Processing (NLP)

---

## Results

- **Accuracy**: Achieved over 85% test accuracy on the validation set.
- **Live Demo**: Users can classify their own Turkish sentences into emotional categories instantly.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/turkish-text-classifier.git
   cd turkish-text-classifier
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app/app.py
   ```

---

## Future Improvements

- Fine-tune a deep learning model (e.g., BERTurk).
- Expand the dataset with more labeled examples.
- Add more emotion classes (like Fear, Disgust, Surprise).

---

## Acknowledgments

- **TREMO Dataset** ([Kaggle Link](https://www.kaggle.com/datasets/mansuralp/tremo)) for providing Turkish emotional text data.

---

## Author

**Işıkay Karakuş**  
- [GitHub](https://github.com/isikaykarakus)  
- [LinkedIn](https://www.linkedin.com/in/isikaykarakus/)  
