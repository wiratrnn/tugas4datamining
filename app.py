import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    silhouette_score,
    davies_bouldin_score,
)
from imblearn.over_sampling import SMOTE

# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="RFM Clustering & Classification App",
    page_icon="📊",
    layout="wide",
)

st.title("RFM Clustering dan Klasifikasi Pelanggan")
st.caption(
    "Aplikasi Streamlit ini mereplikasi alur notebook: preprocessing data, clustering K-Means berbasis RFM, "
    "lalu klasifikasi cluster menggunakan Regresi Logistik dan Naive Bayes."
)

# -----------------------------
# Utility functions
# -----------------------------
DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

REQUIRED_BASE_COLUMNS = [
    "order_id",
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "payment_value",
    "payment_installments",
    "review_score",
    "customer_unique_id",
]


@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    df = pd.read_csv(uploaded_file)
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def build_rfm(df: pd.DataFrame):
    recent = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("customer_unique_id")
        .agg(
            last_purchase=("order_purchase_timestamp", "max"),
            frequency=("order_id", "nunique"),
            monetary=("payment_value", "sum"),
        )
        .reset_index()
    )
    rfm["recency"] = (recent - rfm["last_purchase"]).dt.days
    rfm = rfm[["customer_unique_id", "recency", "frequency", "monetary"]]
    return rfm


def clean_rfm(rfm: pd.DataFrame, max_frequency=10, max_recency=650):
    rfm_clean = rfm[(rfm["frequency"] < max_frequency) & (rfm["recency"] <= max_recency)].copy()
    return rfm_clean


def scale_rfm(rfm_clean: pd.DataFrame):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm_clean[["recency", "frequency", "monetary"]])
    scaled_df = pd.DataFrame(scaled, columns=["recency", "frequency", "monetary"])
    return scaler, scaled_df


def determine_k_metrics(rfm_scaled: pd.DataFrame, k_range=range(2, 7)):
    elbow, silhouette_km, dbi = [], [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=123, n_init=10)
        labels = km.fit_predict(rfm_scaled)
        elbow.append(km.inertia_)
        silhouette_km.append(silhouette_score(rfm_scaled, labels, sample_size=min(10000, len(rfm_scaled)), random_state=123))
        dbi.append(davies_bouldin_score(rfm_scaled, labels))
    return list(k_range), elbow, silhouette_km, dbi


def fit_kmeans(rfm_scaled: pd.DataFrame, n_clusters=2):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(rfm_scaled)
    return model, labels


def prepare_classification_data(df: pd.DataFrame, rfm_clean: pd.DataFrame):
    data_model = df.merge(
        rfm_clean[["customer_unique_id", "cluster"]],
        on="customer_unique_id",
        how="inner",
    ).copy()

    # Coba pakai kolom yang sudah ada, kalau belum ada maka dihitung dari tanggal yang tersedia.
    if "delivery_delay" not in data_model.columns:
        if "aktual_hari" in data_model.columns and "estimasi_hari" in data_model.columns:
            data_model["delivery_delay"] = data_model["aktual_hari"] - data_model["estimasi_hari"]
        else:
            delivered = (data_model["order_delivered_customer_date"] - data_model["order_purchase_timestamp"]).dt.days
            estimated = (data_model["order_estimated_delivery_date"] - data_model["order_purchase_timestamp"]).dt.days
            data_model["delivery_delay"] = delivered - estimated

    features = ["payment_installments", "review_score", "delivery_delay"]
    data_model = data_model.dropna(subset=features + ["cluster"]).copy()
    X = data_model[features]
    y = data_model["cluster"]
    return data_model, X, y, features


def plot_3d_rfm(rfm_clean: pd.DataFrame):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        rfm_clean["recency"],
        rfm_clean["monetary"],
        rfm_clean["frequency"],
        s=10,
        alpha=0.7,
    )
    ax.set_xlabel("Recency")
    ax.set_ylabel("Monetary")
    ax.set_zlabel("Frequency")
    ax.set_title("Sebaran RFM 3D")
    return fig

def plot_elbow_silhouette_dbi(k_values, elbow, silhouette_km, dbi):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(k_values, elbow, marker="o")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Jumlah Cluster (k)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(k_values, silhouette_km, marker="o")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Jumlah Cluster (k)")
    axes[1].set_ylabel("Nilai Silhouette")

    axes[2].plot(k_values, dbi, marker="o")
    axes[2].set_title("Davies-Bouldin Index")
    axes[2].set_xlabel("Jumlah Cluster (k)")
    axes[2].set_ylabel("Nilai DBI")

    plt.tight_layout()
    return fig


def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    return fig


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Input Data")
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

st.sidebar.header("Parameter Clustering")
max_frequency = st.sidebar.slider("Batas frequency outlier", 2, 20, 10)
max_recency = st.sidebar.slider("Batas recency outlier", 100, 1000, 650)
n_clusters = st.sidebar.selectbox("Jumlah cluster", options=[2, 3, 4, 5, 6], index=0)

st.sidebar.header("Parameter Klasifikasi")
test_size = st.sidebar.slider("Proporsi data uji", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

# -----------------------------
# Load and validate data
# -----------------------------
df = load_data(uploaded_file)

if df is None:
    st.info("Silakan upload file CSV dataset terlebih dahulu.")
    st.stop()

missing = [col for col in REQUIRED_BASE_COLUMNS if col not in df.columns]
if missing:
    st.error(
        "Dataset belum lengkap untuk menjalankan alur ini. "
        f"Kolom yang belum ada: {', '.join(missing)}"
    )
    st.stop()

st.subheader("Ringkasan Dataset")
c1, c2, c3 = st.columns(3)
c1.metric("Jumlah baris", f"{len(df):,}")
c2.metric("Jumlah kolom", f"{df.shape[1]}")
c3.metric("Jumlah customer unik", f"{df['customer_unique_id'].nunique():,}")

with st.expander("Tampilkan 5 baris pertama"):
    st.dataframe(df.head(), use_container_width=True)

# -----------------------------
# Preprocessing
# -----------------------------
st.subheader("1) Preprocessing dan RFM")

rfm = build_rfm(df)
rfm_clean = clean_rfm(rfm, max_frequency=max_frequency, max_recency=max_recency)

st.write("RFM awal:")
st.dataframe(rfm.head(), use_container_width=True)

st.write("RFM setelah pembersihan outlier:")
st.dataframe(rfm_clean.head(), use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    pass
    # pairplot_data = rfm_clean[["recency", "frequency", "monetary"]].sample(
    #     n=min(2000, len(rfm_clean)), random_state=42
    # )
    # fig_pair = sns.pairplot(pairplot_data, plot_kws={"alpha": 0.5})
    # st.pyplot(fig_pair.fig, clear_figure=True)
with c2:
    fig_3d = plot_3d_rfm(rfm_clean)
    st.pyplot(fig_3d, clear_figure=True)

scaler_rfm, rfm_scaled = scale_rfm(rfm_clean)

k_values, elbow, silhouette_km, dbi = determine_k_metrics(rfm_scaled, k_range=range(2, 7))
fig_metrics = plot_elbow_silhouette_dbi(k_values, elbow, silhouette_km, dbi)
st.pyplot(fig_metrics, clear_figure=True)

st.caption(
    "Pada notebook asli, k=2 dipilih karena silhouette paling tinggi dan DBI paling rendah. "
    "Di aplikasi ini Anda bisa memilih k dari sidebar, tetapi default-nya tetap 2."
)

# -----------------------------
# Clustering
# -----------------------------
st.subheader("2) Clustering K-Means")
kmeans, cluster_labels = fit_kmeans(rfm_scaled, n_clusters=n_clusters)
rfm_clean = rfm_clean.copy()
rfm_clean["cluster"] = cluster_labels

cluster_profile = (
    rfm_clean.groupby("cluster")
    .agg({"recency": "mean", "frequency": "mean", "monetary": ["mean", "count"]})
    .round(2)
)

st.write("Profil cluster:")
st.dataframe(cluster_profile, use_container_width=True)

st.write("Distribusi cluster:")
st.dataframe(rfm_clean["cluster"].value_counts().sort_index().rename("count"), use_container_width=True)

# -----------------------------
# Classification
# -----------------------------
st.subheader("3) Klasifikasi Cluster")

data_model, X, y, features = prepare_classification_data(df, rfm_clean)

st.write("Fitur klasifikasi yang digunakan:", ", ".join(features))
st.write(f"Ukuran data model: {X.shape[0]:,} baris dan {X.shape[1]} fitur")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=int(random_state),
    stratify=y,
)

scaler_cls = StandardScaler()
X_train_scaled = scaler_cls.fit_transform(X_train)
X_test_scaled = scaler_cls.transform(X_test)

st.write("Distribusi kelas pada data latih sebelum SMOTE:")
st.dataframe(pd.Series(y_train).value_counts().rename("count"), use_container_width=True)

# sm = SMOTE(random_state=int(random_state))
# X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
X_train_res, y_train_res = X_train_scaled, y_train

st.success("SMOTE berhasil diterapkan pada data latih.")

# Logistic Regression
log_model = LogisticRegression(max_iter=1000, random_state=int(random_state))
log_model.fit(X_train_res, y_train_res)
y_pred_log = log_model.predict(X_test_scaled)

# Gaussian Naive Bayes
bayes_model = GaussianNB()
bayes_model.fit(X_train_res, y_train_res)
y_pred_bayes = bayes_model.predict(X_test_scaled)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Regresi Logistik**")
    st.write(f"Akurasi: {accuracy_score(y_test, y_pred_log):.4f}")
    st.text(classification_report(y_test, y_pred_log))
    st.pyplot(plot_confusion(y_test, y_pred_log, "Confusion Matrix - Regresi Logistik"), clear_figure=True)

with col2:
    st.markdown("**Naive Bayes**")
    st.write(f"Akurasi: {accuracy_score(y_test, y_pred_bayes):.4f}")
    st.text(classification_report(y_test, y_pred_bayes))
    st.pyplot(plot_confusion(y_test, y_pred_bayes, "Confusion Matrix - Naive Bayes"), clear_figure=True)

st.caption(
    "Catatan: pada notebook asli, prediksi Naive Bayes sempat menggunakan data uji yang belum diskalakan. "
    "Di versi ini, data uji yang sudah diskalakan digunakan agar alurnya konsisten."
)

# -----------------------------
# Downloadable outputs
# -----------------------------
with st.expander("Lihat data dengan cluster"):
    st.dataframe(rfm_clean.head(100), use_container_width=True)

csv = rfm_clean.to_csv(index=False).encode("utf-8")
st.download_button(
    "Unduh hasil clustering (CSV)",
    data=csv,
    file_name="rfm_clustered.csv",
    mime="text/csv",
)
