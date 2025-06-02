import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("movie_metadata.csv")
        return df
    except FileNotFoundError:
        st.error("File 'movie_metadata.csv' tidak ditemukan.")
        return None

def show_dataset_overview():
    st.title("Dataset Overview")

    df = load_data()
    if df is None:
        return

    # Preprocessing
    df['genres'] = df['genres'].fillna("Unknown")
    df['title_year'] = pd.to_numeric(df['title_year'], errors='coerce')

    st.markdown("### Filter Data")

    # Genre filter
    genre_list = []
    for g in df['genres']:
        genre_list.extend(g.split("|"))
    unique_genres = sorted(set(genre_list))
    st.markdown(
    '<div style="color:#003b49; margin-bottom:-100px;">🎬 Pilih Genre</div>',
    unsafe_allow_html=True
    )
    selected_genre = st.selectbox("", ["Semua"] + unique_genres)

    # Year filter
    years = sorted(df['title_year'].dropna().astype(int).unique())
    st.markdown(
    '<div style="color:#003b49; margin-bottom:-100px;">📅 Pilih Tahun Rilis</div>',
    unsafe_allow_html=True
    )
    selected_year = st.selectbox("", ["Semua"] + years)

    # IMDb score filter
    st.markdown(
    '<div style="color:#003b49; margin-bottom:-100px;">⭐ Pilih Rentang IMDb Score</div>',
    unsafe_allow_html=True
    )
    min_score, max_score = st.slider("", 0.0, 10.0, (0.0, 10.0), step=0.1)

    # Filter dataframe
    filtered_df = df.copy()
    if selected_genre != "Semua":
        filtered_df = filtered_df[filtered_df['genres'].str.contains(selected_genre, na=False)]
    if selected_year != "Semua":
        filtered_df = filtered_df[filtered_df['title_year'] == selected_year]
    filtered_df = filtered_df[(filtered_df['imdb_score'] >= min_score) & (filtered_df['imdb_score'] <= max_score)]

    st.markdown("### Tabel Data Hasil Filter")
    st.dataframe(filtered_df)

    st.markdown("### Visualisasi Dinamis Berdasarkan Filter")

    # 1. Genre Terpopuler - Barplot Interaktif
    st.subheader("1. Genre Terpopuler (Filtered)")
    genre_counts = {}
    for genres in filtered_df['genres']:
        for genre in genres.split('|'):
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

    if genre_counts:
        genre_series = pd.Series(genre_counts).sort_values(ascending=False).head(10).reset_index()
        genre_series.columns = ['Genre', 'Jumlah Film']

        fig1 = px.bar(
            genre_series,
            x='Jumlah Film',
            y='Genre',
            orientation='h',
            title='Top 10 Genre Terpopuler',
            text='Jumlah Film',
            color='Jumlah Film',
            color_continuous_scale='Viridis',
            template='plotly_dark'
        )
        fig1.update_traces(
            textposition='outside',
            textfont_color='#003b49',
            marker_line_width=1.5,
            marker_line_color='rgba(0,0,0,0.8)',
            hovertemplate='<b>%{y}</b><br>Jumlah Film: %{x}<extra></extra>'
        )

        fig1.update_layout(
            title=dict(
            text='Top 10 Genre Terpopuler (Filtered)',
            font=dict(color='#003b49', size=20)
        ),
        coloraxis_colorbar=dict(
            title=dict(text='Jumlah Film', font=dict(color='#003b49')),
            tickfont=dict(color='#003b49'),
            tickcolor='#003b49',
        ),
        yaxis=dict(
            categoryorder='total ascending',
            tickfont=dict(color='#003b49'),       # Warna angka sumbu Y
            titlefont=dict(color='#003b49')       # Warna label sumbu Y ("Genre")
        ),
        xaxis=dict(
            tickfont=dict(color='#003b49'),       # Warna angka sumbu X
            titlefont=dict(color='#003b49')       # Warna label sumbu X ("Jumlah Film")
        ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='black')
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Tidak ada genre cocok setelah filter.")

    st.markdown("### 2. Statistik Deskriptif")
    st.markdown("""
Statistik deskriptif membantu kita memahami sebaran data untuk setiap fitur, termasuk nilai minimum, maksimum, rata-rata (*mean*), standar deviasi, dan lain-lain.
Statistik ini diterapkan pada semua jenis fitur — baik numerik maupun kategorikal.
""")
    st.write(df.describe(include='all'))

    # 2. Heatmap Korelasi Numerik - Interaktif
    st.subheader("3. Korelasi Fitur Numerik")
    if not filtered_df.empty:
        numeric_cols = filtered_df.select_dtypes(include=["int64", "float64"])
        if numeric_cols.shape[1] >= 2:
            corr = numeric_cols.corr()

            fig2 = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                colorbar=dict(
                title="Korelasi",
                titlefont=dict(color='#003b49'),
                tickfont=dict(color='#003b49')
            ),
                hoverongaps=False
            ))
            fig2.update_layout(
            title=dict(
                text='Heatmap Korelasi Fitur Numerik (Filtered)',
                font=dict(color='#003b49', size=20)
            ),
            title_font=dict(color='#003b49'),
            xaxis=dict(
                tickfont=dict(color='#003b49'),
                titlefont=dict(color='#003b49')
            ),
            yaxis=dict(
                tickfont=dict(color='#003b49'),
                titlefont=dict(color='#003b49')
            ),
            template='plotly_dark',  # optional, can be removed if not using dark theme
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='black')  # General font fallback
        )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Tidak cukup fitur numerik untuk korelasi.")
    else:
        st.warning("Data kosong setelah difilter.")
