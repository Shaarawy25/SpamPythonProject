import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from imblearn.over_sampling import SMOTE

# Load data
data = pd.read_csv('sms_spam.csv')
x = data['text']
y = data['type']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english').fit(x_train)
sms_train_vectorized = vectorizer.transform(x_train)
sms_test_vectorized = vectorizer.transform(x_test)

sm = SMOTE(random_state=0)
sms_train_vectorized_resampled, y_train_resampled = sm.fit_resample(sms_train_vectorized, y_train)

clfr = MultinomialNB()
clfr.fit(sms_train_vectorized_resampled, y_train_resampled)

predicted = clfr.predict(sms_test_vectorized)
print(metrics.classification_report(y_test, predicted))

st.title("Spam Classification App")
st.image('SpamIMG.png', use_column_width=True)
st.text('Model Description: Naive Bayes Model, trained on sms data')

text = st.text_input("Enter Text Here", "Type Here...")
predict = st.button('Predict')
if predict:
    new_test_data = vectorizer.transform([text])
    predicted_label = clfr.predict(new_test_data)[0]
    prediction_text = "Spam" if predicted_label == "spam" else "Human"
    if predicted_label == 'spam':
        st.error(f"'{text}' is classified as {prediction_text}")
    else:
        st.success(f"'{text}' is classified as {prediction_text}")
