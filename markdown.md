| Kategori                    | Deskripsi                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dataset**                 | Dataset yang digunakan adalah [Bank Turnover Dataset](https://www.kaggle.com/datasets/barelydedicated/bank-customer-churn-modeling), yang berisi 10.000 baris data nasabah bank dari wilayah geografis yang berbeda (Prancis, Jerman, dan Spanyol). Setiap baris mencakup informasi demografis (jenis kelamin, usia, status pernikahan), informasi keuangan (saldo, pendapatan tahunan, kredit skor), dan perilaku nasabah (jumlah produk, status keanggotaan aktif, dan apakah mereka keluar dari bank atau tidak). Target klasifikasi adalah variabel `Exited`.                                                           |
| **Masalah**                 | Permasalahan utama dalam proyek ini adalah melakukan prediksi apakah seorang nasabah akan meninggalkan (churn) bank atau tetap menjadi pelanggan. Hal ini merupakan permasalahan klasifikasi biner (binary classification), yang secara bisnis penting untuk mempertahankan loyalitas pelanggan. Prediksi churn secara akurat dapat membantu bank untuk menerapkan strategi retensi pelanggan yang lebih efektif dan efisien. Tantangannya meliputi ketidakseimbangan data (data imbalance), korelasi antar fitur, serta pemilihan fitur yang relevan.                                                                      |
| **Solusi Machine Learning** | Solusi yang diusulkan adalah membangun pipeline Machine Learning berbasis TFX (TensorFlow Extended) yang mencakup proses _data ingestion_, _data transformation_, _model training_, _hyperparameter tuning_, _model evaluation_, hingga _model serving_ secara otomatis. Model yang digunakan bertujuan untuk mempelajari pola perilaku churn pelanggan dan memprediksi nasabah yang berisiko churn. Untuk meningkatkan akurasi dan stabilitas model, dilakukan _hyperparameter tuning_ menggunakan KerasTuner.                                                                                                             |
| **Metode Pengolahan**       | Data diproses menggunakan **TensorFlow Transform (TFT)** dalam pipeline TFX, dengan langkah-langkah sebagai berikut:<br>• **Fitur numerik** seperti `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, dan `EstimatedSalary` dinormalisasi menggunakan **z-score scaling** untuk menyamakan skala antar fitur.<br>• **Fitur kategorikal** `Geography` dan `Gender` dikonversi menjadi **indeks integer** yang stabil menggunakan teknik **vocabulary encoding** (`tft.compute_and_apply_vocabulary`).<br>• **Label target** `Exited` diubah ke tipe data `int64` untuk memastikan kompatibilitas dengan fungsi loss klasifikasi.<br>• Semua hasil transformasi diberi akhiran **`_xf`** agar terpisah dari data mentah dan mudah dilacak sepanjang pipeline.<br>• Proses ini juga mengasumsikan telah dilakukan **validasi awal** untuk memastikan tidak ada nilai `null` dan semua kolom memiliki format yang sesuai sebelum masuk ke tahap transformasi dan training. |
| **Arsitektur Model**        | Model ini dirancang untuk menyelesaikan permasalahan **klasifikasi biner** (nasabah churn atau tidak), dengan memanfaatkan **fitur numerik** dan **fitur kategorikal** dari dataset pelanggan bank. Arsitektur dirancang modular dan ringan, tetapi cukup representatif untuk data tabular.

### **1. Input Layer**

Model menerima dua jenis input:

* **Fitur Numerik** (dtype: `float32`, shape: `(1,)` per fitur):

  * `Age`, `Balance`, `CreditScore`, `EstimatedSalary`, `NumOfProducts`, `Tenure`, `HasCrCard`, `IsActiveMember`
  * Semua fitur ini diasumsikan telah ditransformasikan (misalnya: normalisasi z-score) dan langsung dapat digunakan oleh jaringan.

* **Fitur Kategorikal** (dtype: `int64`, shape: `(1,)` per fitur):

  * `Gender`, `Geography`
  * Fitur ini telah dikonversi menjadi indeks integer dari proses vocabulary encoding sebelumnya.

Setiap fitur dimasukkan sebagai **input tensor individual** dalam model (`tf.keras.Input`), total ada 10 input.

---

### **2. Embedding Layer (untuk fitur kategorikal)**

* **`Gender`**:

  * Memiliki 2 kategori (male/female) → ukuran vocabulary = 2
  * Ditambah 1 untuk bucket OOV → `input_dim=3`
  * Dibentuk embedding vector berdimensi **4** → `output_dim=4`

* **`Geography`**:

  * Memiliki 3 kategori (France, Germany, Spain) → ukuran vocabulary = 3
  * Ditambah 1 untuk bucket OOV → `input_dim=4`
  * Embedding vector berdimensi **4**

* Output embedding diratakan (`Reshape`) menjadi vektor `(4,)` agar dapat digabungkan dengan fitur numerik.

**Tujuan**: memungkinkan model mempelajari representasi laten dari kategori secara lebih fleksibel dan kompak daripada one-hot encoding.

---

### **3. Feature Concatenation**

* Semua fitur numerik (`float32`) dan hasil embedding dari fitur kategorikal digabung menggunakan `layers.Concatenate()`.
* Hasilnya adalah vektor gabungan berdimensi tetap, sebagai representasi lengkap dari satu data pelanggan.

---

### **4. Hidden Layers**

Struktur hidden layer adalah:

1. **Dense(64, ReLU)**
2. **Dense(32, ReLU)**
3. **Dropout(0.3)**
4. **Dense(16, ReLU)**

Desain ini memberikan:

* Kapasitas cukup untuk mempelajari interaksi fitur,
* Regularisasi ringan melalui Dropout (30%) untuk mengurangi risiko overfitting,
* Ukuran yang efisien untuk data tabular dan skala dataset (10.000 baris).

---

### **5. Output Layer**

* **Dense(1, sigmoid)**:

  * Menghasilkan probabilitas antara 0 dan 1.
  * Digunakan untuk klasifikasi biner (churn atau tidak churn).

---

### **6. Loss Function & Optimizer**

* **Loss**: `binary_crossentropy`

  * Cocok untuk target `Exited` yang bernilai 0 atau 1.
* **Optimizer**: `Adam(learning_rate=0.001)`

  * Optimizer adaptif, cocok untuk training cepat dan stabil tanpa banyak tuning manual.

---

### **7. Compile & Build**

Model dikompilasi dengan metrik yang lengkap:

```python
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=[
        BinaryAccuracy(name='accuracy'),
        AUC(name='auc'),
        Precision(name='precision'),
        Recall(name='recall')
    ]
)
```

Model kemudian dicetak dengan `.summary()` untuk memastikan struktur.

---

### **8. Signature Serving**

* Model disiapkan untuk serving dengan `serve_tf_examples_fn`, yang:

  * Menerima data dalam format `tf.Example` (serialized).
  * Melakukan parsing dan transformasi kembali menggunakan `tf_transform_output`.
  * Mengembalikan prediksi langsung dari model.
* Signature ini digunakan dalam `SavedModel` yang disimpan untuk deployment.                                                         |
| **Metrik Evaluasi**         | Evaluasi model dilakukan menggunakan beberapa metrik klasifikasi biner:<br>• **AUC (Area Under ROC Curve):** Metrik utama untuk melihat performa model dalam membedakan kelas churn dan non-churn. <br>• **Accuracy:** Mengukur proporsi prediksi benar dari total prediksi. <br>• **Precision & Recall:** Digunakan untuk memahami keseimbangan antara false positives dan false negatives. <br>• **Confusion Matrix:** Untuk visualisasi klasifikasi dan kesalahan model.                                                                                                                                                 |
| **Performa Model**          | Setelah dilakukan hyperparameter tuning dan pelatihan model, diperoleh hasil sebagai berikut: <br>• **AUC tertinggi pada data validasi** mencapai 0.86, menunjukkan model cukup baik dalam membedakan nasabah yang churn dan yang tidak. <br>• **Accuracy** berada di kisaran 80-82%, tergantung pada kombinasi parameter. <br>• Hyperparameter terbaik diperoleh dari tuner dengan kombinasi 3 hidden layer (64-32-16 units), learning rate 0.001, dan dropout 0.3. <br>• Model akhir dievaluasi dan diberkati (blessed) oleh komponen Evaluator, dan siap untuk didorong ke tahap deployment menggunakan komponen Pusher. |
