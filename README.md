# Diagnosis Penyakit Berdasarkan Gejala Menggunakan MLP, Backpropagation, dan Severity Weighting

Project ini membangun model klasifikasi penyakit berdasarkan daftar gejala menggunakan **Multi Layer Perceptron (MLP)** yang diimplementasikan dari awal menggunakan NumPy. Model dilatih dengan proses **forward propagation**, **backpropagation**, **mini-batch gradient descent**, serta membandingkan beberapa konfigurasi hidden layer, fungsi aktivasi, dan jenis representasi input.

Dataset yang digunakan berasal dari Kaggle: **Disease-Symptom Description Dataset**.

---

## Anggota Kelompok

| Nama | NIM |
|---|---|
| Kevin Febrian Widhiarta | 24031554025 |
| Muhammad Tyas Subianto | 24031554077 |
| Bima Sena | 24031554214 |
| Daris Ikhwana Khoir Suhaya | 24031554215 |

---

## Tujuan Project

Tujuan utama project ini adalah membuat sistem prediksi penyakit berdasarkan gejala dengan pendekatan neural network dari scratch. Secara khusus, project ini mencakup:

- Melakukan preprocessing dataset gejala dan penyakit.
- Mengubah daftar gejala menjadi representasi numerik menggunakan multi-hot encoding.
- Memberikan bobot severity pada setiap gejala.
- Membangun model MLP secara manual menggunakan NumPy.
- Melakukan training menggunakan backpropagation.
- Membandingkan beberapa arsitektur hidden layer.
- Membandingkan fungsi aktivasi ReLU, Sigmoid, dan Tanh.
- Mengevaluasi performa model menggunakan accuracy, precision, recall, dan F1-score.
- Membandingkan performa input berbasis severity weighting dengan input binary.
- Membuat fungsi prediksi penyakit berdasarkan gejala yang dimasukkan secara manual.

---

## Dataset

Dataset diambil menggunakan library `kagglehub` dari:

```text
itachi9604/disease-symptom-description-dataset
```

File yang digunakan:

| File | Keterangan |
|---|---|
| `dataset.csv` | Berisi data penyakit dan daftar gejala pada setiap penyakit |
| `Symptom-severity.csv` | Berisi bobot severity untuk setiap gejala |

Dataset memiliki:

- 4.920 sampel data
- 41 kelas penyakit
- 131 gejala unik setelah preprocessing
- 17 kolom gejala pada data awal

---

## Library yang Digunakan

Project ini menggunakan beberapa library berikut:

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
kagglehub
ipywidgets
```

Library utama untuk implementasi MLP adalah **NumPy**, sedangkan Scikit-learn digunakan untuk preprocessing, splitting data, evaluasi metrik, dan model baseline pembanding.

---

## Alur Project

### 1. Import Library

Notebook memulai proses dengan mengimpor library untuk manipulasi data, visualisasi, preprocessing, evaluasi model, serta model pembanding.

### 2. Load Dataset

Dataset dimuat langsung dari Kaggle menggunakan `kagglehub`, yaitu:

- `dataset.csv`
- `Symptom-severity.csv`

### 3. Exploratory Data Analysis

EDA dilakukan untuk memahami karakteristik dataset, meliputi:

- Distribusi jumlah sampel per penyakit
- Persentase missing values pada kolom gejala
- Distribusi bobot severity gejala
- Top 20 gejala yang paling sering muncul

### 4. Preprocessing Data

Tahapan preprocessing yang dilakukan:

1. Membersihkan nama gejala:
   - Menghapus whitespace
   - Mengubah huruf menjadi lowercase
   - Mengganti spasi menjadi underscore
   - Menghapus underscore ganda

2. Membuat daftar gejala unik.

3. Melakukan **multi-hot encoding**, yaitu mengubah daftar gejala menjadi vektor biner:
   - `1` jika gejala muncul
   - `0` jika gejala tidak muncul

4. Melakukan **label encoding** pada target penyakit.

5. Menerapkan **severity weighting** dengan mengalikan vektor biner dengan bobot severity setiap gejala.

6. Melakukan scaling global pada fitur weighted berdasarkan bobot severity maksimum.

7. Membagi data menjadi:
   - Train set: 3.148 sampel
   - Validation set: 788 sampel
   - Test set: 984 sampel

---

## Arsitektur Model

Model utama adalah **Multi Layer Perceptron** yang dibuat dari awal menggunakan NumPy.

Komponen model:

- Hidden layer fleksibel
- Aktivasi hidden layer:
  - ReLU
  - Sigmoid
  - Tanh
- Output layer menggunakan Softmax
- Loss function menggunakan Categorical Cross-Entropy
- Optimizer menggunakan Mini-batch Gradient Descent
- Regularisasi menggunakan Dropout
- Early stopping menggunakan validation loss

---

## Eksperimen Hidden Layer

Beberapa konfigurasi hidden layer diuji menggunakan input severity weighted dan aktivasi ReLU.

| Eksperimen | Arsitektur | Validation Accuracy | Validation F1-Score |
|---|---:|---:|---:|
| A_1layer_128 | `[131, 128, 41]` | 0.4975 | 0.4112 |
| B_2layer_128-64 | `[131, 128, 64, 41]` | 0.4429 | 0.3603 |
| C_3layer_256-128-64 | `[131, 256, 128, 64, 41]` | 0.6091 | 0.5228 |
| D_4layer_512-256-128-64 | `[131, 512, 256, 128, 64, 41]` | 0.7208 | 0.6482 |

Konfigurasi terbaik berdasarkan validation F1-score adalah:

```text
D_4layer_512-256-128-64
Arsitektur: [131, 512, 256, 128, 64, 41]
```

---

## Eksperimen Fungsi Aktivasi

Setelah arsitektur terbaik dipilih, dilakukan perbandingan fungsi aktivasi ReLU, Sigmoid, dan Tanh.

| Fungsi Aktivasi | Validation Accuracy | Validation Precision | Validation Recall | Validation F1-Score |
|---|---:|---:|---:|---:|
| Tanh | 0.9251 | 0.9083 | 0.9251 | 0.9055 |
| ReLU | 0.7208 | 0.6207 | 0.7208 | 0.6482 |
| Sigmoid | 0.0241 | 0.0006 | 0.0241 | 0.0012 |

Fungsi aktivasi terbaik adalah:

```text
Tanh
```

---

## Evaluasi Final Model

Model terbaik menggunakan:

```text
Arsitektur: [131, 512, 256, 128, 64, 41]
Aktivasi  : Tanh
```

Hasil evaluasi pada test set:

| Metrik | Test Score |
|---|---:|
| Accuracy | 0.9197 |
| Precision | 0.9014 |
| Recall | 0.9197 |
| F1-Score | 0.8962 |

---

## Perbandingan Sebelum dan Sesudah Backpropagation

Project ini juga membandingkan performa model sebelum dan sesudah proses training menggunakan backpropagation.

| Kondisi Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Sebelum Backpropagation | 0.0457 | 0.0132 | 0.0457 | 0.0195 |
| Sesudah Backpropagation | 0.9197 | 0.9014 | 0.9197 | 0.8962 |

Hasil ini menunjukkan bahwa proses backpropagation sangat berpengaruh dalam meningkatkan kemampuan model mengenali pola hubungan antara gejala dan penyakit.

---

## Analisis Kesalahan Prediksi

Pada test set, model menghasilkan:

```text
Jumlah prediksi salah: 79 dari 984 data test
Persentase kesalahan : 8.03%
```

Beberapa pola kesalahan yang paling sering muncul:

| Actual Disease | Predicted Disease | Jumlah Kesalahan |
|---|---|---:|
| Acne | Psoriasis | 22 |
| Malaria | Typhoid | 20 |
| Chronic cholestasis | hepatitis A | 18 |
| Paralysis (brain hemorrhage) | Gastroenteritis | 5 |
| Fungal infection | Drug Reaction | 5 |

---

## Perbandingan Severity Weighted dan Binary Input

Project ini membandingkan performa model dengan input berbasis severity weighting dan input binary biasa.

| Input Type | Val Accuracy | Val F1-Score | Test Accuracy | Test F1-Score |
|---|---:|---:|---:|---:|
| Severity Weighted | 0.9251 | 0.9055 | 0.9197 | 0.8962 |
| Binary | 0.9987 | 0.9987 | 0.9980 | 0.9980 |

Berdasarkan hasil tersebut, input binary menghasilkan performa yang lebih tinggi dibandingkan severity weighted pada dataset ini. Hal ini dapat terjadi karena pola kemunculan gejala pada dataset sudah cukup kuat untuk membedakan kelas penyakit, sehingga bobot severity tidak selalu meningkatkan performa klasifikasi.

---

## Fitur Prediksi Manual

Notebook menyediakan fungsi untuk melakukan prediksi penyakit berdasarkan input gejala manual:

```python
predict_disease_from_symptoms(symptoms, model=best_model, top_k=5, use_severity=True)
```

Contoh penggunaan:

```python
symptoms = ["itching", "skin_rash", "nodal_skin_eruptions"]

predict_disease_from_symptoms(
    symptoms=symptoms,
    model=best_model,
    top_k=5,
    use_severity=True
)
```

Fungsi tersebut akan menghasilkan daftar prediksi penyakit dengan probabilitas tertinggi.

Notebook juga menyediakan antarmuka sederhana menggunakan `ipywidgets`, sehingga pengguna dapat memilih gejala dari dropdown dan menjalankan prediksi secara interaktif.

---

## Cara Menjalankan Project

### 1. Clone Repository

```bash
git clone https://github.com/username/nama-repository.git
cd nama-repository
```

### 2. Install Dependency

```bash
pip install numpy pandas matplotlib seaborn scikit-learn kagglehub ipywidgets
```

### 3. Jalankan Notebook

Buka file notebook di Jupyter Notebook, JupyterLab, atau Google Colab:

```bash
jupyter notebook Source_Code.ipynb
```

Jika menggunakan Google Colab, upload notebook ke Colab lalu jalankan setiap cell secara berurutan.

---

## Struktur Project

Contoh struktur repository:

```text
.
├── Source_Code.ipynb
├── README.md
└── requirements.txt
```

Jika ingin membuat `requirements.txt`, isi file tersebut dapat berupa:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
kagglehub
ipywidgets
```

---

## Visualisasi yang Dihasilkan

Notebook menghasilkan beberapa visualisasi, antara lain:

- Distribusi sampel per penyakit
- Persentase missing values pada kolom gejala
- Distribusi bobot severity gejala
- Top 20 gejala paling sering muncul
- Training loss dan accuracy untuk setiap arsitektur
- Perbandingan metrik validation antar hidden layer
- Perbandingan fungsi aktivasi
- Perbandingan sebelum dan sesudah backpropagation
- Confusion matrix model final
- Perbandingan severity weighted input dan binary input

---

## Kesimpulan

Project ini berhasil membangun model MLP dari scratch untuk melakukan klasifikasi penyakit berdasarkan gejala. Model terbaik menggunakan arsitektur `[131, 512, 256, 128, 64, 41]` dengan fungsi aktivasi Tanh dan memperoleh test accuracy sebesar **91,97%** serta test F1-score sebesar **89,62%** pada input severity weighted.

Eksperimen tambahan menunjukkan bahwa proses backpropagation meningkatkan performa model secara signifikan. Namun, perbandingan dengan input binary menunjukkan bahwa representasi binary menghasilkan performa yang lebih tinggi pada dataset ini, dengan test accuracy sebesar **99,80%** dan test F1-score sebesar **99,80%**.

---

## Catatan

Model ini hanya digunakan untuk pembelajaran machine learning dan eksplorasi metode klasifikasi. Untuk diagnosis penyakit yang sebenarnya, konsultasikan dengan tenaga medis profesional.
