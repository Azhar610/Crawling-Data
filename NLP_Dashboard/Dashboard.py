import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from textblob import TextBlob
import io

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Stopwords untuk bahasa Indonesia
stopwords_indonesia = set(stopwords.words('indonesian'))

# Buat stemmer bahasa Indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Setup halaman Streamlit
st.title("NLP Dashboard dengan Dataset Pribadi")

# Upload Dataset
uploaded_file = st.file_uploader("Upload dataset Anda (format CSV)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Membaca dataset
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.write("Data yang diupload:")
    st.write(df.head())  # Tampilkan beberapa baris pertama

    # Pilih kolom teks
    column_name = st.selectbox("Pilih kolom teks untuk analisis:", df.columns)

    if st.button("Analyze"):
        st.subheader("Hasil Analisis NLP")

        # Tokenisasi
        df['Tokens'] = df[column_name].apply(lambda x: word_tokenize(str(x)))
        st.write("Tokenisasi:")
        st.write(df[['Tokens']].head())

        # Hapus stopwords
        df['Filtered Tokens'] = df['Tokens'].apply(lambda x: [word for word in x if word.lower() not in stopwords_indonesia])
        st.write("Token setelah Stopwords dihapus:")
        st.write(df[['Filtered Tokens']].head())

        # Stemming
        df['Stemmed Tokens'] = df['Filtered Tokens'].apply(lambda x: [stemmer.stem(word) for word in x])
        st.write("Stemming:")
        st.write(df[['Stemmed Tokens']].head())

        # Analisis Sentimen (opsional untuk bahasa Inggris)
        df['Sentiment Polarity'] = df[column_name].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        df['Sentiment Subjectivity'] = df[column_name].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
        st.write("Analisis Sentimen (Polarity & Subjectivity):")
        st.write(df[['Sentiment Polarity', 'Sentiment Subjectivity']].head())

        # Opsi download hasil analisis
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download hasil analisis sebagai CSV",
            data=csv,
            file_name='hasil_analisis_nlp.csv',
            mime='text/csv',
        )
