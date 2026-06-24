# Machine Learning — UAS & Midterm Projects

Repositori ini berisi dua notebook implementasi end-to-end pipeline Machine Learning
untuk tugas **UAS** dan **Midterm** mata kuliah Machine Learning.

Setiap notebook mencakup pipeline lengkap mulai dari pemuatan data, eksplorasi,
preprocessing, pelatihan model, hyperparameter tuning dengan Optuna,
hingga evaluasi dan tracking eksperimen dengan MLflow.

---

## Identifikasi

| Field | Detail |
|-------|--------|
| Nama | Najwa Bilqis Al Khalidah |
| NIM | 101032300186 |
| Kelas | TK-46-GAB |
| Mata Kuliah | Machine Learning |

---

## Deskripsi Proyek

Repositori ini terdiri dari dua proyek utama:

1. **UAS — Fraud Detection Pipeline** (`1_Online_Transaction_UAS_ML.ipynb`): Klasifikasi transaksi online sebagai fraud atau non-fraud menggunakan berbagai algoritma ML dengan penanganan class imbalance.
2. **Midterm — Regression Pipeline** (`2_regression_pipeline.ipynb`): Prediksi tahun rilis lagu menggunakan arsitektur Deep Learning (PyTorch) dengan tuning otomatis dan interpretasi model.

Kedua notebook mencakup:
- Eksplorasi data (EDA) dan analisis distribusi fitur
- Preprocessing, feature engineering, dan penanganan missing values
- Hyperparameter tuning otomatis dengan Optuna
- Tracking eksperimen menggunakan MLflow
- Visualisasi hasil evaluasi model

---

## Struktur Repository

```
finalterm-machine-learning/
|
|-- 1_Online_Transaction_UAS_ML.ipynb
|-- 2_regression_pipeline.ipynb
|-- README.md
```

---

## Deskripsi Setiap Notebook

### Notebook 1: UAS — End-to-End Fraud Detection Pipeline
**File:** `1_Online_Transaction_UAS_ML.ipynb`

Implementasi pipeline machine learning untuk mendeteksi transaksi online yang bersifat
penipuan (fraud). Dataset yang digunakan adalah `train_transaction.csv` yang diunduh
dari Google Drive menggunakan `gdown`.

Topik utama:
- Pengunduhan dataset dari Google Drive dan eksplorasi awal data
- Analisis distribusi label target yang sangat tidak seimbang (~3.5% fraud)
- Analisis missing values dan korelasi fitur numerik dengan target
- Preprocessing: penghapusan kolom dengan missing >50%, Label Encoding fitur kategorikal
- Feature engineering: log transformation TransactionAmt, fitur waktu (jam, is_night, hari)
- Penanganan class imbalance dengan SMOTE
- Pelatihan empat model: Logistic Regression, Random Forest, XGBoost, LightGBM
- Hyperparameter tuning LightGBM dengan Optuna (10 trial)
- Tracking eksperimen dengan MLflow
- Evaluasi: ROC-AUC, F1 Score, Precision, Recall, Confusion Matrix
- Visualisasi ROC Curve dan Precision-Recall Curve untuk semua model

---

### Notebook 2: Midterm — End-to-End Regression Pipeline (Deep Learning)
**File:** `2_regression_pipeline.ipynb`

Implementasi pipeline regresi berbasis Deep Learning untuk memprediksi
tahun rilis lagu dari fitur-fitur audio. Dataset berupa file CSV tanpa header
dengan kolom pertama sebagai target (`release_year`).

Topik utama:
- Pemuatan dan penamaan kolom dataset tanpa header
- EDA: distribusi target tahun rilis dan korelasi fitur dengan target
- Preprocessing: imputasi missing values dengan median, filter outlier IQR (1–99 percentile)
- Train-val-test split 70/15/15 dengan normalisasi fitur (StandardScaler) dan target
- Arsitektur deep learning dengan `RegressionNet`: Linear + BatchNorm + ReLU + Dropout
- Hyperparameter tuning dengan Optuna: jumlah layer, ukuran hidden unit, dropout, learning rate, batch size, weight decay (20 trial)
- Pelatihan model final dengan Adam optimizer, ReduceLROnPlateau scheduler, dan early stopping
- Tracking metrik per epoch dan model artifact dengan MLflow
- Evaluasi: MSE, RMSE, MAE, R2 Score; plot Prediksi vs Aktual dan Distribusi Residual
- Interpretasi model dengan LIME: top-10 fitur paling berpengaruh per sampel

---

## Library yang Digunakan

| Library | Kegunaan |
|---------|----------|
| Python 3.8+ | Bahasa pemrograman utama |
| scikit-learn | Preprocessing, model klasifikasi, evaluasi |
| pandas | Manipulasi data tabular |
| numpy | Operasi array dan matematika |
| matplotlib / seaborn | Visualisasi |
| imbalanced-learn | SMOTE untuk class imbalance |
| xgboost | Model XGBoost |
| lightgbm | Model LightGBM |
| torch (PyTorch) | Arsitektur dan pelatihan deep learning |
| optuna | Hyperparameter tuning otomatis |
| mlflow | Experiment tracking dan model logging |
| lime | Interpretasi model (lokal) |
| gdown | Unduh dataset dari Google Drive |

---

## Cara Menjalankan

### Prasyarat

**Notebook 1 (Fraud Detection):**
```bash
pip install gdown optuna mlflow lightgbm xgboost imbalanced-learn scikit-learn pandas numpy matplotlib seaborn
```

**Notebook 2 (Regression Pipeline):**
```bash
pip install torch optuna lime mlflow scikit-learn pandas numpy matplotlib seaborn
```

### Menjalankan Notebook

```bash
git clone https://github.com/njwbilll/finalterm-machine-learning.git
cd finalterm-machine-learning
jupyter notebook
```

Kemudian buka notebook sesuai kebutuhan di browser.

### Catatan Dataset

- **Notebook 1:** Dataset `train_transaction.csv` diunduh otomatis dari Google Drive saat cell pertama dijalankan. Tidak perlu download manual.
- **Notebook 2:** Letakkan file `midterm-regresi-dataset.csv` di direktori yang sama dengan notebook sebelum menjalankan.

---

## Dataset yang Digunakan

| Dataset | Notebook | Deskripsi |
|---------|----------|-----------|
| train_transaction.csv | 1 (UAS) | Dataset transaksi online, target biner `isFraud`, sangat tidak seimbang |
| midterm-regresi-dataset.csv | 2 (Midterm) | Dataset audio lagu, target numerik `release_year` (tahun rilis) |

---

## Hasil dan Kesimpulan

### Notebook 1 — Fraud Detection
- SMOTE berhasil menyeimbangkan distribusi kelas dan meningkatkan recall kelas fraud.
- LightGBM dan XGBoost memberikan performa terbaik karena kemampuan gradient boosting menangkap pola kompleks.
- Metrik utama yang digunakan: ROC-AUC dan Average Precision, karena dataset sangat tidak seimbang.
- Optuna menemukan kombinasi hyperparameter LightGBM optimal dalam 10 trial.

### Notebook 2 — Regression Pipeline
- Model `RegressionNet` dengan BatchNorm dan Dropout mampu memprediksi tahun rilis lagu dengan rata-rata error beberapa tahun.
- Optuna secara otomatis menemukan arsitektur (jumlah layer, ukuran hidden, learning rate) terbaik dalam 20 trial.
- LIME memberikan interpretasi lokal yang menunjukkan fitur audio mana yang paling memengaruhi prediksi per sampel.
- Seluruh metrik (MSE, RMSE, MAE, R2) dan artifact divisualisasikan dan dicatat melalui MLflow.
