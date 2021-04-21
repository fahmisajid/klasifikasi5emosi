import pandas as pd
import streamlit as st

import pickle 

st.title("Prediksi Emosi")

st.markdown(""" ## Judul Penelitian: Perbandingan Algoritma Klasifikasi terhadap Emosi Tweet Berbahasa Indonesia
Dataset: [Indonesian-Twitter-Emotion-Dataset](https://github.com/meisaputri21/Indonesian-Twitter-Emotion-Dataset)  \n
Text Preprocessing: _ Cleaning _, _ Case Folding _, Normalisasi  \n
Algoritma: Logistic Regtression, Hyperparameter C = 1.623776739188721  \n

""")


df = pd.read_csv('Twitter_Emotion_Datasetab.csv')
pkl_filename = "Lrpickle_model.pkl"

with open(pkl_filename, 'rb') as file:
    classifier = pickle.load(file)

with open("countVectLR", 'rb') as file:
    count_vect = pickle.load(file)

with open("tfidfLR", 'rb') as file:
    tfidf_transformer = pickle.load(file)

sentence = st.text_input('Masukkan Kalimat:') 

text_new =[sentence]
X_new_counts = count_vect.transform(text_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

prediction = classifier.predict(X_new_tfidf)
prediction_proba = classifier.predict_proba(X_new_tfidf)

if sentence:
    st.text("Emosi:")
    st.write(prediction[0])

    st.subheader('Kelas Label dan Nomor Indeks')
    st.write(classifier.classes_)

    st.subheader('Probabilitas Prediksi')
    st.write(prediction_proba)

