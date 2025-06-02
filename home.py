import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("movie_metadata.csv")
        return df
    except FileNotFoundError:
        st.error("File 'movie_metadata.csv' tidak ditemukan.")
        return None

def show_home():
    st.title("Movie Recommendation Dashboard")

    # CSS styling untuk tema cadetblue-putih dan metric box
    st.markdown("""
    <style>
    /* Judul dan teks utama */
    h1, h2, h3, h4, h5, h6, p {
        color: #003b49;
    }

    /* Horizontal line jadi hitam */
    hr {
        border: 1px solid black;
    }

    /* Metric styling */
    [data-testid="stMetric"] {
        background-color: #e0f7fa;  /* Light cyan (biru muda) */
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #80cbc4;  /* cadetblue terang */
        text-align: center;
    }
    [data-testid="stMetric"] > div {
        color: #003b49 !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""  
    Selamat datang di **Dashboard Rekomendasi Film!**  
    Dashboard ini dirancang untuk membantu Anda menganalisis data film secara menyeluruh. Dimulai dari eksplorasi dataset, proses preprocessing, hingga penerapan algoritma clustering. Hasil analisis ini memberikan insight serta rekomendasi film berdasarkan karakteristik data yang dimiliki.
    """)

    st.markdown('<hr style="border: 1px solid black;">', unsafe_allow_html=True)

    
    st.subheader("Tujuan Proyek")
    st.markdown("""
    - Membangun sistem rekomendasi film berbasis **clustering**.  
    - Membantu pengguna menemukan film dengan karakteristik yang **serupa**.  
    - Menerapkan metode **K-Means** dan **DBSCAN** untuk segmentasi film.  
    """)

    st.subheader("Dataset")
    st.markdown("""
    - Dataset: **IMDB 5000 Movie Dataset**  
    - Jumlah data: **5043 film**  
    - Atribut penting: *genre*, *imdb_score*, *duration*, *budget*, *actor*, *director*  
    """)

    st.markdown('<hr style="border: 1px solid black;">', unsafe_allow_html=True)

    st.subheader("Fitur Utama Dashboard")
    st.markdown("""
    - Menyediakan eksplorasi data film berdasarkan genre, tahun rilis, dan skor IMDb.
    - Menampilkan visualisasi grafis untuk analisis genre populer dan korelasi antar fitur numerik.
    - Menyajikan hasil clustering menggunakan algoritma **K-Means** dan **DBSCAN**.
    - Memberikan interpretasi hasil clustering untuk mendukung sistem rekomendasi film. 
    - Melampirkan poster akademik sebagai ringkasan visual dari keseluruhan proyek.
    """)

    df = load_data()
    if df is None:
        return

    # Statistik ringkas dataset
    st.subheader("Statistik Singkat Dataset")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jumlah Film", df.shape[0])
    with col2:
        st.metric("Jumlah Fitur", df.shape[1])
    with col3:
        unique_genres = set()
        for g in df['genres'].fillna('Unknown'):
            unique_genres.update(g.split("|"))
        st.metric("Jumlah Genre Unik", len(unique_genres))

    # Grafik distribusi jumlah film per tahun
    st.subheader("Distribusi Jumlah Film per Tahun Rilis")

    df_year = df.dropna(subset=['title_year'])
    year_counts = df_year['title_year'].astype(int).value_counts().sort_index()

    df_plot = pd.DataFrame({
    'Tahun Rilis': year_counts.index,
    'Jumlah Film': year_counts.values
    })

    fig_year = px.bar(
    df_plot,
    x='Tahun Rilis',
    y='Jumlah Film',
    title="Jumlah Film per Tahun Rilis",
    template='plotly_white',
    opacity=0.85,
    color_discrete_sequence=["#008b8b"]
    )

    fig_year.update_layout(
    xaxis=dict(
        dtick=5,
        tickfont=dict(color="#003b49"),
        titlefont=dict(color="#003b49")
    ),
    yaxis=dict(
        tickfont=dict(color="#003b49"),
        titlefont=dict(color="#003b49")
    ),
    title_text="Jumlah Film per Tahun Rilis",
    title_font_size=18,
    title_font_color="#003b49",
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color="#003b49")
)

    st.plotly_chart(fig_year, use_container_width=True)

    st.markdown('<hr style="border: 1px solid black;">', unsafe_allow_html=True)

    st.markdown("Dashboard ini dibuat untuk memberikan pemahaman yang lebih dalam serta mendukung pengambilan keputusan dalam merekomendasikan film berdasarkan data yang tersedia. Gunakan menu navigasi di samping untuk menjelajahi fitur-fitur lainnya.")
