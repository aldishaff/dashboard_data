import streamlit as st

st.set_page_config(page_title="Dashboard Rekomendasi Film", layout="wide")

# CSS Styling Biru & Putih (CadetBlue)
st.markdown("""
<style>
/* Sidebar background dan teks */
[data-testid="stSidebar"] {
    background-color: #d1e7f0 ; /* Biru sangat muda */
    color: #5f9ea0 !important;
}

/* Judul sidebar & label */
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] h4, 
[data-testid="stSidebar"] h5, 
[data-testid="stSidebar"] h6, 
[data-testid="stSidebar"] label, 
[data-testid="stSidebar"] p {
    color: #003b49 !important;
    font-weight: 600;
}

/* Radio button teks */
.css-1d391kg, .css-1cpxqw2, .stRadio label {
    color: #5f9ea0 !important;
    font-weight: 600 !important;
}

/* Radio button aktif */
.stRadio [role="radio"][aria-checked="true"] {
    background-color: #b2d8d8;
    border-radius: 5px;
    color: #5f9ea0 !important;
}

/* Judul dan konten utama */
h1, h2, h3, h4, h5, h6 {
    color: #5f9ea0;
    font-weight: 700;
}
body, .stApp {
    background-color: #ffffff;
    color: #1e3f4f; /* Biru tua untuk kontras teks */
    font-family: "Segoe UI", sans-serif;
}

/* Tombol */
.stButton>button {
    background-color: #5f9ea0;
    color: #ffffff;
    border-radius: 8px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: 600;
}
.stButton>button:hover {
    background-color: #3d6e70;
    color: #ffffff;
}

/* Box peringatan */
.stAlert {
    background-color: #e0f7fa;
    border-left: 5px solid #5f9ea0;
    color: #1e3f4f;
}

/* Footer */
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #5f9ea0;
    color: #ffffff;
    text-align: center;
    padding: 15px 10px;
    font-size: 0.9rem;
    z-index: 9999;
}
.footer a {
    color: #003b49;
    text-decoration: none;
}
.footer a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# Sidebar navigasi
st.sidebar.title("Navigasi Dashboard")
menu = st.sidebar.radio("Pilih Halaman", [
    "Home",
    "Dataset",
    # "Preprocessing & EDA",
    "K-Means",
    "DBSCAN",
    "Hasil",
    "Poster"
])

# Inisialisasi session state
if 'kmeans_result' not in st.session_state:
    st.session_state.kmeans_result = None
if 'dbscan_result' not in st.session_state:
    st.session_state.dbscan_result = None

# Navigasi halaman
if menu == "Home":
    from home import show_home
    show_home()

elif menu == "Dataset":
    from dataset_overview import show_dataset_overview
    show_dataset_overview()

# elif menu == "Preprocessing & EDA":
#     from preprocessing_eda import show_preprocessing_eda
#     show_preprocessing_eda()

elif menu == "K-Means":
    from clustering_kmeans import show_clustering_kmeans
    st.session_state.kmeans_result = show_clustering_kmeans()

elif menu == "DBSCAN":
    from clustering_dbscan import show_clustering_dbscan
    st.session_state.dbscan_result = show_clustering_dbscan(st.session_state.df, st.session_state.df_clustering)

elif menu == "Hasil":
    from interpretasi import show_interpretasi
    show_interpretasi()

elif menu == "Poster":
    from poster import show_poster_film
    show_poster_film()

# Footer tetap di bawah
st.markdown("""
<div class="footer">
    <strong>Disusun oleh:</strong><br>
    Aulia Nadya Celviana – 202210370311382 | <a href="mailto:auliacelviana678@webmail.umm.ac.id">auliacelviana678@webmail.umm.ac.id</a><br>
    Larasati Khadijah Kalimantari Karnain – 202210370311410 | <a href="mailto:larasatikhadijah@wemail.umm.ac.id">larasatikhadijah@wemail.umm.ac.id</a>
</div>
""", unsafe_allow_html=True)
