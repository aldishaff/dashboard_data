import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go

def show_clustering_kmeans():
    st.title("Clustering K-Means")

    # --- Load data ---
    try:
        df = pd.read_csv("movie_metadata.csv")
    except FileNotFoundError:
        st.error("File 'movie_metadata.csv' tidak ditemukan.")
        return None

    # --- Preprocessing: numeric only, drop ID column if exists ---
    df_numeric = df.select_dtypes(include=np.number).copy()
    df_numeric.drop(columns=['movie_id'], errors='ignore', inplace=True)
    df_numeric.dropna(inplace=True)

    # --- Split train/test ---
    train_data_kmeans, test_data_kmeans = train_test_split(df_numeric, test_size=0.2, random_state=42)
    scaler_kmeans = StandardScaler()
    train_scaled_kmeans = scaler_kmeans.fit_transform(train_data_kmeans)
    test_scaled_kmeans = scaler_kmeans.transform(test_data_kmeans)

    # --- 1. Tentukan K optimal dari data training ---
    range_k = range(2, 11)
    silhouette_scores_train = []
    for k in range_k:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(train_scaled_kmeans)
        score = silhouette_score(train_scaled_kmeans, kmeans.labels_)
        silhouette_scores_train.append(score)

    n_clusters_optimal = range_k[np.argmax(silhouette_scores_train)]
    st.markdown("## A. Clustering K-Means dengan Original Data")

    # --- Visualisasi 1: Silhouette Score Training (Plotly) ---
    st.subheader("1. Grafik Silhouette Score Clustering K-Means dengan Data Training")
    fig1 = px.line(
        x=list(range_k), y=silhouette_scores_train,
        markers=True,
        labels={'x': 'Jumlah Cluster (K)', 'y': 'Silhouette Score'},
        title='Silhouette Score untuk Menentukan K Optimal (Data Training)'
    )
    fig1.add_vline(x=n_clusters_optimal, line_dash='dash', line_color='red')
    
    fig1.update_layout(
    title=dict(
        text='Silhouette Score untuk Menentukan K Optimal (Data Training)',
        font=dict(color='#003b49')
    ),
    xaxis=dict(
        title='Jumlah Cluster (K)',
        titlefont=dict(color='#003b49'),
        tickfont=dict(color='#003b49'),
        showgrid=False,
        zeroline=False
    ),
    yaxis=dict(
        title='Silhouette Score',
        titlefont=dict(color='#003b49'),
        tickfont=dict(color='#003b49'),
        showgrid=True,
        zeroline=False
    ),
    plot_bgcolor='rgba(0,0,0,0)',  # transparan
    paper_bgcolor='rgba(0,0,0,0)',  # transparan
)
    st.plotly_chart(fig1)

    # 🔍 Deskripsi Hasil 1
    st.markdown(
        f"""
        **Hasilnya:** Jumlah cluster optimal ditentukan dengan melihat puncak tertinggi pada silhouette score, 
        yang dalam hal ini terjadi pada **cluster ke-{n_clusters_optimal}**.
        """
    )

    # --- Latih KMeans dengan K optimal ---
    kmeans_final = KMeans(n_clusters=n_clusters_optimal, random_state=42, n_init=10)
    kmeans_final.fit(train_scaled_kmeans)
    test_labels = kmeans_final.predict(test_scaled_kmeans)

    # --- Visualisasi 2: Silhouette Plot Testing ---
    st.subheader("2. Grafik Silhouette Score untuk Clustering K-Means dengan Data Testing")
    silhouette_vals = silhouette_samples(test_scaled_kmeans, test_labels)
    silhouette_df = pd.DataFrame({
        'silhouette': silhouette_vals,
        'cluster': test_labels
    })
    silhouette_df.sort_values(by=['cluster', 'silhouette'], inplace=True)
    silhouette_df['index'] = np.arange(len(silhouette_df))

    fig2 = px.bar(
        silhouette_df, x='silhouette', y='index',
        color='cluster', orientation='h',
        labels={'silhouette': 'Silhouette Coefficient', 'index': 'Sample Index'},
        title=f'Silhouette Plot (Testing) - K = {n_clusters_optimal}'
    )
    fig2.add_vline(x=silhouette_df['silhouette'].mean(), line_dash='dash', line_color='red')
    fig2.update_layout(
    title=dict(
        text=f'Silhouette Plot (Testing) - K = {n_clusters_optimal}',
        font=dict(color='#003b49')
    ),
    xaxis=dict(
        title='Silhouette Coefficient',
        titlefont=dict(color='#003b49'),
        tickfont=dict(color='#003b49'),
        showgrid=False,
        zeroline=False
    ),
    yaxis=dict(
        title='Sample Index',
        titlefont=dict(color='#003b49'),
        tickfont=dict(color='#003b49'),
        showticklabels=False,
        showgrid=False,
        zeroline=False
    ),
    legend=dict(
        font=dict(color='#003b49')
    ),
    plot_bgcolor='rgba(0,0,0,0)',  # transparan
    paper_bgcolor='rgba(0,0,0,0)'  # transparan
)
    st.plotly_chart(fig2)

    # 🔍 Deskripsi Hasil 2
    avg_silhouette = round(silhouette_df['silhouette'].mean(), 3)
    st.markdown(
        f"""
        **Hasilnya:** Silhouette plot menunjukkan bahwa hasil clustering dengan **{n_clusters_optimal} cluster**
        cukup baik dengan rata-rata nilai silhouette adalah **{avg_silhouette}** atau sekitar **{round(avg_silhouette, 1)}**.
        """
    )

    # --- Visualisasi 3: PCA Clustering ---
    st.subheader("3. Grafik Clustering K-Means dengan Data Testing")

    if test_scaled_kmeans.shape[1] > 2:
        pca = PCA(n_components=2)
        test_pca = pca.fit_transform(test_scaled_kmeans)
        centroids_pca = pca.transform(kmeans_final.cluster_centers_)
    else:
        test_pca = test_scaled_kmeans
        centroids_pca = kmeans_final.cluster_centers_

    pca_df = pd.DataFrame({
        'PCA1': test_pca[:, 0],
        'PCA2': test_pca[:, 1],
        'cluster': test_labels
    })

    fig3 = px.scatter(
        pca_df, x='PCA1', y='PCA2', color=pca_df['cluster'].astype(str),
        title=f'Clustering K-Means pada Data Testing (K={n_clusters_optimal})',
        labels={'cluster': 'Cluster'},
        opacity=0.7
    )
    fig3.add_trace(go.Scatter(
        x=centroids_pca[:, 0], y=centroids_pca[:, 1],
        mode='markers',
        marker=dict(symbol='x', color='black', size=12, line=dict(width=2, color='red')),
        name='Centroids'
    ))
    fig3.update_layout(
    title=dict(
        text=f'Clustering K-Means pada Data Testing (K={n_clusters_optimal})',
        font=dict(color='#003b49')
    ),
    xaxis=dict(
        title='PCA1',
        titlefont=dict(color='#003b49'),
        tickfont=dict(color='#003b49'),
        showgrid=False,
        zeroline=False
    ),
    yaxis=dict(
        title='PCA2',
        titlefont=dict(color='#003b49'),
        tickfont=dict(color='#003b49'),
        showgrid=True,
        zeroline=False
    ),
    legend=dict(
        font=dict(color='#003b49')
    ),
    plot_bgcolor='rgba(0,0,0,0)',  # transparan
    paper_bgcolor='rgba(0,0,0,0)'  # transparan
)
    st.plotly_chart(fig3)

    # 🔍 Deskripsi Hasil 3
    st.markdown(
        """
        **Hasilnya:** Visualisasi menunjukkan bahwa data terbagi menjadi dua cluster, 
        dengan cluster ungu yang lebih rapat dan jelas, sedangkan cluster kuning lebih menyebar 
        dan berada di tengah. Ini menghasilkan pemisahan yang cukup baik namun belum sempurna, 
        yang sejalan dengan nilai silhouette sekitar 0.3.
        """
    )





    # --- B. Clustering With K-Means Feature Selection ---
    st.markdown("## B. Clustering K-Means dengan Feature Selection")

    # ===== Preprocessing =====
    df = df[['movie_title', 'genres', 'duration', 'budget', 'imdb_score',
            'actor_1_name', 'actor_2_name', 'actor_3_name', 'director_name']].copy()

    df['duration'].fillna(df['duration'].mean(), inplace=True)
    df['budget'].fillna(df['budget'].mean(), inplace=True)
    df['imdb_score'].fillna(df['imdb_score'].mean(), inplace=True)
    df.fillna("Unknown", inplace=True)

    le = LabelEncoder()
    df['actor_1_encoded'] = le.fit_transform(df['actor_1_name'])
    df['actor_2_encoded'] = le.fit_transform(df['actor_2_name'])
    df['actor_3_encoded'] = le.fit_transform(df['actor_3_name'])
    df['director_encoded'] = le.fit_transform(df['director_name'])
    df['genre_main'] = df['genres'].str.split('|').str[0]
    df['genre_main_encoded'] = le.fit_transform(df['genre_main'])

    # Ambil fitur yang telah diseleksi untuk clustering
    df_clustering = df[['genre_main_encoded', 'imdb_score', 'duration', 'budget',
                        'actor_1_encoded', 'actor_2_encoded', 'actor_3_encoded', 'director_encoded']]

    # --- Split train/test ---
    train_data_kmeans, test_data_kmeans = train_test_split(df_clustering, test_size=0.2, random_state=42)
    scaler_kmeans = StandardScaler()
    train_scaled_kmeans = scaler_kmeans.fit_transform(train_data_kmeans)
    test_scaled_kmeans = scaler_kmeans.transform(test_data_kmeans)

    # --- 1. Tentukan K optimal dari data training ---
    st.subheader("1. Grafik Silhouette Score Clustering K-Means dengan Data Training (Feature Selection)")
    range_k = range(2, 11)
    silhouette_scores_train = []
    for k in range_k:
        kmeans_train_silhouette = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_train_silhouette.fit(train_scaled_kmeans)
        silhouette_avg_train = silhouette_score(train_scaled_kmeans, kmeans_train_silhouette.labels_)
        silhouette_scores_train.append(silhouette_avg_train)

    fig1 = px.line(
        x=list(range_k), y=silhouette_scores_train,
        markers=True,
        labels={'x': 'Jumlah Cluster (K)', 'y': 'Silhouette Score'},
        title='Silhouette Score untuk Menentukan Jumlah Cluster Optimal (Data Training)'
    )
    n_clusters_optimal_kmeans = range_k[np.argmax(silhouette_scores_train)]
    fig1.add_vline(x=n_clusters_optimal_kmeans, line_dash='dash', line_color='red')

    fig1.update_layout(
    title=dict(
        text='Silhouette Score untuk Menentukan Jumlah Cluster Optimal (Data Training)',
        font=dict(color='#003b49')
    ),
    xaxis=dict(
        title='Jumlah Cluster (K)',
        titlefont=dict(color='#003b49'),
        tickfont=dict(color='#003b49'),
        showgrid=False,
        zeroline=False
    ),
    yaxis=dict(
        title='Silhouette Score',
        titlefont=dict(color='#003b49'),
        tickfont=dict(color='#003b49'),
        showgrid=True,
        zeroline=False
    ),
    plot_bgcolor='rgba(0,0,0,0)',    # transparan
    paper_bgcolor='rgba(0,0,0,0)'    # transparan
)
    st.plotly_chart(fig1)

    st.markdown(f"**Hasilnya:** Jumlah cluster optimal berdasarkan silhouette score data training adalah **{n_clusters_optimal_kmeans}**.")

    # --- Latih KMeans dengan K optimal ---
    kmeans_train = KMeans(n_clusters=n_clusters_optimal_kmeans, random_state=42, n_init=10)
    kmeans_train.fit(train_scaled_kmeans)
    train_labels_kmeans = kmeans_train.predict(train_scaled_kmeans)
    test_labels_kmeans = kmeans_train.predict(test_scaled_kmeans)

    # st.markdown("### Distribusi Cluster")
    # st.write("Distribusi cluster pada data training:")
    # st.write(pd.Series(train_labels_kmeans).value_counts().sort_index())

    # st.write("Distribusi cluster pada data testing:")
    # st.write(pd.Series(test_labels_kmeans).value_counts().sort_index())

    # --- 2. Visualisasi Silhouette Plot Data Testing ---
    st.subheader(f"2. Grafik Silhouette Plot untuk {n_clusters_optimal_kmeans} Cluster (Data Testing)")
    silhouette_vals = silhouette_samples(test_scaled_kmeans, test_labels_kmeans)
    silhouette_df = pd.DataFrame({
        'silhouette': silhouette_vals,
        'cluster': test_labels_kmeans
    })
    silhouette_df.sort_values(by=['cluster', 'silhouette'], inplace=True)
    silhouette_df['index'] = np.arange(len(silhouette_df))

    fig2 = px.bar(
        silhouette_df, x='silhouette', y='index',
        color='cluster', orientation='h',
        labels={'silhouette': 'Silhouette Coefficient', 'index': 'Sample Index'},
        title=f'Silhouette Plot (Testing) - K = {n_clusters_optimal_kmeans}',
        height=600
    )
    avg_silhouette_test = round(silhouette_df['silhouette'].mean(), 3)
    fig2.add_vline(x=avg_silhouette_test, line_dash='dash', line_color='red')
    fig2.update_layout(
    title=dict(
        text='Silhouette Plot (Testing) - K = {n_clusters_optimal_kmeans}',
        font=dict(color='#003b49')
    ),
    xaxis=dict(
        title='Jumlah Cluster (K)',
        titlefont=dict(color='#003b49'),
        tickfont=dict(color='#003b49'),
        showgrid=False,
        zeroline=False
    ),
    yaxis=dict(
        title='Silhouette Score',
        titlefont=dict(color='#003b49'),
        tickfont=dict(color='#003b49'),
        showgrid=False,
        zeroline=False
    ),
    plot_bgcolor='rgba(0,0,0,0)',   # transparan
    paper_bgcolor='rgba(0,0,0,0)'   # transparan
)
    st.plotly_chart(fig2)

    st.markdown(f"**Hasilnya:** Silhouette Score sebesar **{avg_silhouette_test}** menunjukkan bahwa pemisahan antar cluster masih kurang baik dan data cenderung saling tumpang tindih antar cluster.")

    # --- 3. Visualisasi Clustering dengan PCA ---
    st.subheader(f"3. Visualisasi Clustering K-Means pada Data Testing dengan PCA (K={n_clusters_optimal_kmeans})")
    if train_scaled_kmeans.shape[1] > 2:
        pca = PCA(n_components=2)
        test_pca = pca.fit_transform(test_scaled_kmeans)
        centroids_pca = pca.transform(kmeans_train.cluster_centers_)
    else:
        test_pca = test_scaled_kmeans
        centroids_pca = kmeans_train.cluster_centers_

    pca_df = pd.DataFrame({
        'PCA1': test_pca[:, 0],
        'PCA2': test_pca[:, 1],
        'cluster': test_labels_kmeans.astype(str)
    })

    fig3 = px.scatter(
        pca_df, x='PCA1', y='PCA2', color='cluster',
        title=f'Clustering K-Means pada Data Testing (K={n_clusters_optimal_kmeans})',
        labels={'cluster': 'Cluster'},
        opacity=0.7
    )
    fig3.add_trace(go.Scatter(
        x=centroids_pca[:, 0], y=centroids_pca[:, 1],
        mode='markers',
        marker=dict(symbol='x', color='black', size=12, line=dict(width=2, color='red')),
        name='Centroids'
    ))
    fig3.update_layout(
    title=dict(
        text=f'Clustering K-Means pada Data Testing (K={n_clusters_optimal_kmeans})',
        font=dict(color='#003b49')
    ),
    xaxis=dict(
        title='PCA1',
        titlefont=dict(color='#003b49'),
        tickfont=dict(color='#003b49'),
        showgrid=False,
        zeroline=False
    ),
    yaxis=dict(
        title='PCA2',
        titlefont=dict(color='#003b49'),
        tickfont=dict(color='#003b49'),
        showgrid=True,
        zeroline=False
    ),
    legend=dict(
        font=dict(color='#003b49')
    ),
    plot_bgcolor='rgba(0,0,0,0)',  # transparan
    paper_bgcolor='rgba(0,0,0,0)'  # transparan
)
    st.plotly_chart(fig3)

    st.markdown("**Hasilnya:** Pembagian 9 cluster terlalu banyak karena sebagian besar data saling tumpang tindih, dan ada satu data yang jauh sendiri kemungkinan outlier.")

    return df_numeric, n_clusters_optimal, df_clustering, n_clusters_optimal_kmeans
