import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    df = pd.read_csv("movie_metadata.csv")
    return df

def show_preprocessing_eda():
    st.title("🧪 Preprocessing & EDA")

    # Link ke Google Colab
    st.markdown("[📎 Klik di sini untuk membuka versi Google Colab](https://colab.research.google.com/drive/1WPCuLHlCU-1yEevb9aBDLPSUDzKpQ-9n?usp=sharing)", unsafe_allow_html=True)

    df = load_data()

    st.markdown("## 🗂️ Exploratory Data Analysis (EDA)")
    st.markdown("""
Tahapan ini bertujuan untuk memahami struktur, isi, serta karakteristik dari data film sebelum masuk ke tahap preprocessing dan modeling. Dengan EDA, kita dapat mengidentifikasi pola, outlier, missing value, serta karakteristik penting dari fitur yang ada.
""")

    st.markdown("### 🔍 Struktur dan Tipe Data")
    st.markdown("""
Langkah pertama adalah memahami dimensi dan struktur data. Kita perlu mengetahui berapa banyak baris (record) dan kolom (fitur) dalam dataset, serta tipe data tiap kolom.
Informasi ini penting untuk menentukan bagaimana fitur-fitur tersebut akan diproses nantinya, seperti apakah perlu encoding, scaling, atau imputasi.
""")
    st.write(f"Dataset memiliki **{df.shape[0]} baris** dan **{df.shape[1]} kolom**.")
    st.write("**Tipe Data Tiap Kolom:**")
    st.write(df.dtypes)
    st.write("**Contoh 5 Baris Pertama Dataset:**")
    st.write(df.head())

    st.markdown("### 📊 Statistik Deskriptif")
    st.markdown("""
Statistik deskriptif membantu kita memahami sebaran data untuk setiap fitur, termasuk nilai minimum, maksimum, rata-rata (*mean*), standar deviasi, dan lain-lain.
Statistik ini diterapkan pada semua jenis fitur — baik numerik maupun kategorikal.
""")
    st.write(df.describe(include='all'))

    st.markdown("### ❗ Pengecekan Missing Values")
    st.markdown("""
Salah satu aspek penting dalam EDA adalah menangani data yang hilang (*missing values*). Kita perlu mengetahui berapa banyak data yang hilang di setiap kolom, karena nilai kosong dapat memengaruhi akurasi model.
""")
    st.write("**Jumlah missing values di setiap kolom:**")
    st.write(df.isnull().sum())

    st.markdown("### 📌 Missing Values pada Kolom Kunci")
    st.markdown("""
Berikut ini adalah kolom-kolom penting yang umum digunakan dalam analisis atau pemodelan, seperti `duration`, `budget`, `imdb_score`, dan nama-nama aktor serta sutradara.
""")
    cols_to_check = ['duration', 'budget', 'imdb_score', 
                     'actor_1_name', 'actor_2_name', 
                     'actor_3_name', 'director_name']
    st.write(df[cols_to_check].isnull().sum())

    st.markdown("### 🛠️ Penanganan Missing Value (Numerik)")
    st.markdown("""
Untuk atribut numerik seperti `duration`, `budget`, dan `imdb_score`, kita dapat mengisi nilai yang hilang dengan nilai rata-rata kolom tersebut.
Pendekatan ini sederhana namun efektif dalam banyak kasus untuk menghindari kehilangan informasi karena penghapusan data.
""")
    df['duration'] = df['duration'].fillna(df['duration'].mean())
    df['budget'] = df['budget'].fillna(df['budget'].mean())
    df['imdb_score'] = df['imdb_score'].fillna(df['imdb_score'].mean())
    st.write("**Contoh data setelah imputasi nilai hilang (numerik):**")
    st.write(df[['duration', 'budget', 'imdb_score']].head())

    st.success("✅ EDA dan preprocessing selesai dilakukan. Dataset siap untuk tahap selanjutnya seperti clustering atau sistem rekomendasi.")
