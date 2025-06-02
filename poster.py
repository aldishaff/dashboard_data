import streamlit as st

def show_poster_film():
    st.title("Poster Film & Link Akses")

    st.markdown("""
    Halaman ini menampilkan **Academic Poster** yang dibuat dengan tujuan menyajikan informasi atau hasil penelitian secara visual dan ringkas.  
    Poster akademik ini dirancang untuk memudahkan pemahaman konsep, data, dan hasil studi melalui kombinasi teks, grafik, dan ilustrasi yang menarik.  
    """)
    # Contoh data poster: judul, path file lokal poster, dan link Canva
    posters = [
        {
            "title": "Academic Poster",
            "poster_path": "images/Poster Data Minning.png",  # path lokal file poster
            "link": "https://www.canva.com/design/DAGm7sTNj3k/0xYQ9mSyHJke-rLNO7umqA/view"  # link Canva
        },
    ]

    for movie in posters:
        st.subheader(movie["title"])
        st.image(movie["poster_path"], use_container_width=True)
        st.markdown(f"[🔗 Lihat Poster di Canva]({movie['link']})", unsafe_allow_html=True)
        st.markdown("---")
