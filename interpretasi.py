import streamlit as st


def show_interpretasi():

    st.title("Hasil dan Kesimpulan Analisis Clustering")

    st.markdown("""
    ### Perbandingan Hasil Clustering:
    - Pada algoritma **K-Means**, data original menghasilkan clustering yang lebih baik dengan **Silhouette Score sebesar 0.299** dan hanya 2 cluster, dibandingkan dengan feature selection yang menghasilkan 9 cluster dan Silhouette Score lebih rendah, yaitu **0.116**.
    - Pada algoritma **DBSCAN**, feature selection memberikan hasil yang lebih baik dengan jumlah cluster yang lebih banyak dan noise yang lebih sedikit dibandingkan data original. Data original cenderung menghasilkan noise yang sangat tinggi, terutama pada data testing, di mana seluruh data masuk ke dalam cluster -1 (noise).

    ### Kesimpulan Analisis Clustering
    - **K-Means Clustering**  
    Skenario data original memberikan hasil terbaik dengan Silhouette Score 0.299, sedangkan feature selection menghasilkan performa terendah dengan skor 0.116.  
    Penurunan ini cukup signifikan, artinya seleksi fitur justru menurunkan kualitas pemisahan cluster pada K-Means.

    - **DBSCAN Clustering**  
    Pada data training, original data menghasilkan banyak noise (data dianggap tidak masuk cluster), sementara feature selection membentuk lebih banyak cluster yang lebih jelas.  
    Pada data testing, original data hampir seluruhnya dianggap noise, sedangkan feature selection menghasilkan beberapa cluster yang bisa dikenali.

    - **Pengaruh Feature Selection**  
    Pada K-Means, seleksi fitur menurunkan performa. Namun pada DBSCAN, seleksi fitur justru membantu membentuk cluster yang lebih baik dan mengurangi noise.

    - **Masalah Noise di DBSCAN**  
    Noise tinggi pada DBSCAN (khususnya di original data) bisa terjadi karena parameter yang kurang optimal atau distribusi data testing yang berbeda dari data training.
    """)
