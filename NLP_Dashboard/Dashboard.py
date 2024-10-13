import streamlit as st
import pandas as pd
import spacy

# Load model bahasa multibahasa
nlp = spacy.load("xx_ent_wiki_sm")

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

        # Tokenisasi dan penghapusan stopwords
        def process_text(text):
            # Pastikan input adalah string
            if isinstance(text, str):
                doc = nlp(text)
                tokens = [token.text for token in doc if not token.is_stop]
                return tokens
            else:
                return []  # Kembalikan list kosong jika input bukan string

        df['Tokens'] = df[column_name].apply(process_text)
        st.write("Tokenisasi (tanpa stopwords):")
        st.write(df[['Tokens']].head())

        # Lemmatization menggunakan spaCy
        def lemmatize_tokens(tokens):
            if tokens:  # Pastikan tokens tidak kosong
                return [token.lemma_ for token in nlp(" ".join(tokens))]
            return []  # Kembalikan list kosong jika tokens kosong

        df['Lemmatized Tokens'] = df['Tokens'].apply(lemmatize_tokens)
        st.write("Lemmatized Tokens:")
        st.write(df[['Lemmatized Tokens']].head())
