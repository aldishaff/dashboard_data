import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_clustering_dbscan(df, df_clustering):
    if 'df' not in st.session_state or 'df_clustering' not in st.session_state:
        st.error("Data df dan df_clustering harus diberikan.")
        return

    st.title("Clustering DBSCAN")

    # === A. Clustering With DBSCAN Algorithm Original Data ===
    st.header("A. Clustering DBSCAN with Original Data")

    df_original_processed_numeric = df.select_dtypes(include=np.number).copy()
    cols_to_drop_full = ['movie_id']
    df_original_processed_numeric = df_original_processed_numeric.drop(columns=cols_to_drop_full, errors='ignore')
    df_original_processed_numeric = df_original_processed_numeric.dropna()

    train_data_full, test_data_full = train_test_split(df_original_processed_numeric, test_size=0.2, random_state=42)

    scaler_full = StandardScaler()
    train_scaled_full = scaler_full.fit_transform(train_data_full)
    test_scaled_full = scaler_full.transform(test_data_full)

    eps_optimal_train_full = 1.5
    min_samples_optimal_train_full = 5

    dbscan_train_full = DBSCAN(eps=eps_optimal_train_full, min_samples=min_samples_optimal_train_full)
    train_labels_full = dbscan_train_full.fit_predict(train_scaled_full)
    test_labels_full = dbscan_train_full.fit_predict(test_scaled_full)

    # st.write(f"Jumlah cluster pada data training (full features): {len(np.unique(train_labels_full))}")
    # st.write("Distribusi cluster pada data training (full features):")
    # st.write(pd.Series(train_labels_full).value_counts())

    # st.write(f"Jumlah cluster pada data testing (full features): {len(np.unique(test_labels_full))}")
    # st.write("Distribusi cluster pada data testing (full features):")
    # st.write(pd.Series(test_labels_full).value_counts())

    if train_scaled_full.shape[1] >= 2:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Training Data", "Testing Data"))
        unique_labels_train_full = np.unique(train_labels_full)
        colors_train_full = px.colors.qualitative.Plotly
        for i, k in enumerate(unique_labels_train_full):
            color = 'black' if k == -1 else colors_train_full[i % len(colors_train_full)]
            class_member_mask = (train_labels_full == k)
            xy = train_scaled_full[class_member_mask, :2]
            fig.add_trace(go.Scatter(x=xy[:, 1], y=xy[:, 0],
                                    mode='markers',
                                    marker=dict(color=color, size=6, opacity=0.5),
                                    name=f'Cluster {k} (Train)'),
                        row=1, col=1)

        unique_labels_test_full = np.unique(test_labels_full)
        for i, k in enumerate(unique_labels_test_full):
            color = 'black' if k == -1 else colors_train_full[i % len(colors_train_full)]
            class_member_mask = (test_labels_full == k)
            xy = test_scaled_full[class_member_mask, :2]
            fig.add_trace(go.Scatter(x=xy[:, 1], y=xy[:, 0],
                                    mode='markers',
                                    marker=dict(color=color, size=6, opacity=0.5),
                                    name=f'Cluster {k} (Test)'),
                        row=1, col=2)

        # Update X and Y axes - left plot (Training)
        fig.update_xaxes(title_text=df_original_processed_numeric.columns[1] + ' (Scaled)',
                        color='black',  # warna angka ticks
                        tickfont=dict(color='black'),
                        title_font=dict(color='black'),  # warna label sumbu X
                        showline=False,  # sembunyikan garis sumbu
                        row=1, col=1)
        fig.update_yaxes(title_text=df_original_processed_numeric.columns[0] + ' (Scaled)',
                        color='black',  # warna angka ticks
                        tickfont=dict(color='black'),
                        title_font=dict(color='black'),  # warna label sumbu Y
                        showline=False,  # sembunyikan garis sumbu
                        row=1, col=1)

        # Update X and Y axes - right plot (Testing)
        fig.update_xaxes(title_text=df_original_processed_numeric.columns[1] + ' (Scaled)',
                        color='black',
                        tickfont=dict(color='black'),
                        title_font=dict(color='black'),
                        showline=False,
                        row=1, col=2)
        fig.update_yaxes(title_text=df_original_processed_numeric.columns[0] + ' (Scaled)',
                        color='black',
                        tickfont=dict(color='black'),
                        title_font=dict(color='black'),
                        showline=False,
                        row=1, col=2)


        fig.update_layout(title=dict(
        text="Clustering pada Data Training & Testing",
        font=dict(color='#003b49')  # Warna hijau tua
        ),
                        legend=dict(title="Cluster", x=1.02, y=1, font=dict(color='black')),
                        height=500, width=900,
                        plot_bgcolor='white', paper_bgcolor='white',
                        font=dict(color='black'))

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
Scatter plot dibuat untuk menampilkan hasil clustering pada dua fitur pertama, baik untuk data training maupun testing.
- **Hasilnya:**
    - Pada data **training**, algoritma DBSCAN berhasil membentuk beberapa cluster berbeda.
    - Pada data **testing**, semua data dimasukkan ke **Cluster -1**, yang menandakan bahwa data testing dianggap sebagai **noise** atau **tidak cocok** dengan cluster yang telah terbentuk dari data training.
""")

    # === B. Clustering With DBSCAN Algorithm Feature Selection ===
    st.header("B. Clustering DBSCAN with Feature Selection")

    train_data_selected, test_data_selected = train_test_split(df_clustering, test_size=0.2, random_state=42)

    scaler_selected = StandardScaler()
    train_scaled_selected = scaler_selected.fit_transform(train_data_selected)
    test_scaled_selected = scaler_selected.transform(test_data_selected)

    eps_optimal_train_selected = 1.0
    min_samples_optimal_train_selected = 5

    dbscan_train_selected = DBSCAN(eps=eps_optimal_train_selected, min_samples=min_samples_optimal_train_selected)
    train_labels_selected = dbscan_train_selected.fit_predict(train_scaled_selected)
    test_labels_selected = dbscan_train_selected.fit_predict(test_scaled_selected)

    # st.write(f"Jumlah cluster pada data training (fitur terseleksi): {len(np.unique(train_labels_selected))}")
    # st.write("Distribusi cluster pada data training (fitur terseleksi):")
    # st.write(pd.Series(train_labels_selected).value_counts())

    # st.write(f"Jumlah cluster pada data testing (fitur terseleksi): {len(np.unique(test_labels_selected))}")
    # st.write("Distribusi cluster pada data testing (fitur terseleksi):")
    # st.write(pd.Series(test_labels_selected).value_counts())

    if train_scaled_selected.shape[1] >= 2:
        fig_b = make_subplots(rows=1, cols=2, subplot_titles=("Training Data", "Testing Data"))
        unique_labels_train_selected = np.unique(train_labels_selected)
        colors_selected = px.colors.qualitative.Plotly
        for i, k in enumerate(unique_labels_train_selected):
            color = 'black' if k == -1 else colors_selected[i % len(colors_selected)]
            class_member_mask = (train_labels_selected == k)
            xy = train_scaled_selected[class_member_mask, :2]
            fig_b.add_trace(go.Scatter(x=xy[:,0], y=xy[:,1],
                                    mode='markers',
                                    marker=dict(color=color, size=6, opacity=0.5),
                                    name=f'Cluster {k} (Train)'),
                            row=1, col=1)

        unique_labels_test_selected = np.unique(test_labels_selected)
        for i, k in enumerate(unique_labels_test_selected):
            color = 'black' if k == -1 else colors_selected[i % len(colors_selected)]
            class_member_mask = (test_labels_selected == k)
            xy = test_scaled_selected[class_member_mask, :2]
            fig_b.add_trace(go.Scatter(x=xy[:,0], y=xy[:,1],
                                    mode='markers',
                                    marker=dict(color=color, size=6, opacity=0.5),
                                    name=f'Cluster {k} (Test)'),
                            row=1, col=2)

        # Plot kiri (Train)
        fig_b.update_xaxes(
            title_text='genre_main_encoded (Scaled)',  # Pastikan kolom sesuai
            tickfont=dict(color='black'),
            title_font=dict(color='black'),
            showline=False,
            row=1, col=1
        )
        fig_b.update_yaxes(
            title_text='imdb_score (Scaled)',
            tickfont=dict(color='black'),
            title_font=dict(color='black'),
            showline=False,
            row=1, col=1
        )

        # Plot kanan (Test)
        fig_b.update_xaxes(
            title_text='genre_main_encoded (Scaled)',
            tickfont=dict(color='black'),
            title_font=dict(color='black'),
            showline=False,
            row=1, col=2
        )
        fig_b.update_yaxes(
            title_text='imdb_score (Scaled)',
            tickfont=dict(color='black'),
            title_font=dict(color='black'),
            showline=False,
            row=1, col=2
        )

        fig_b.update_layout(
            title=dict(
        text="Clustering pada Data Training & Testing",
        font=dict(color='#003b49')  # Warna hijau tua
        ),
            legend=dict(title="Cluster", x=1.02, y=1, font=dict(color='black')),
            height=500, width=900,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black')
        )
        st.plotly_chart(fig_b, use_container_width=True)

        st.markdown("""
Scatter plot untuk data training dan testing menampilkan **cluster berwarna** dan **noise berwarna hitam**. Plot juga dilengkapi dengan judul, label sumbu, dan legenda untuk mempermudah identifikasi.
- **Hasilnya:**
    - **Grafik kiri (data training)**: Data pelatihan berhasil dikelompokkan ke dalam **26 cluster berbeda (Cluster 0–25)**. Hal ini menunjukkan adanya **variasi yang besar** dalam pola fitur (genre, skor IMDb, dll).
    - **Grafik kanan (data testing)**: Sebagian besar data testing berhasil masuk ke **cluster yang sama** dengan hasil training (**Cluster 0–10**), namun terdapat **sebagian kecil data** yang dimasukkan ke **Cluster -1**, menunjukkan bahwa data tersebut tidak sesuai dengan cluster manapun (noise).
""")
        return {
        'train_labels_full': train_labels_full,
        'test_labels_full': test_labels_full,
        'train_labels_selected': train_labels_selected,
        'test_labels_selected': test_labels_selected
    }


# ===== Main script =====
if 'df' not in st.session_state:
    df = pd.read_csv("movie_metadata.csv")
    df.fillna({'duration': df['duration'].mean(),
               'budget': df['budget'].mean(),
               'imdb_score': df['imdb_score'].mean()}, inplace=True)
    df.fillna("Unknown", inplace=True)

    le = LabelEncoder()
    df['actor_1_encoded'] = le.fit_transform(df['actor_1_name'].astype(str))
    df['actor_2_encoded'] = le.fit_transform(df['actor_2_name'].astype(str))
    df['actor_3_encoded'] = le.fit_transform(df['actor_3_name'].astype(str))
    df['director_encoded'] = le.fit_transform(df['director_name'].astype(str))
    df['genre_main'] = df['genres'].str.split('|').str[0].fillna("Unknown")
    df['genre_main_encoded'] = le.fit_transform(df['genre_main'])

    st.session_state.df = df

if 'df_clustering' not in st.session_state:
    df_clustering = st.session_state.df[['genre_main_encoded', 'imdb_score', 'duration', 'budget',
                                        'actor_1_encoded', 'actor_2_encoded', 'actor_3_encoded', 'director_encoded']]
    st.session_state.df_clustering = df_clustering
