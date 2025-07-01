# 🐾 Proyek Klasifikasi Gambar Wajah Hewan - Animal Faces

## 📌 Pendahuluan

Proyek ini menggunakan dataset [Animal Faces](https://www.kaggle.com/datasets/alessiocorrado99/animal-faces) dari Kaggle yang berisi **15.716 gambar** wajah hewan yang telah dipotong dan disejajarkan. Gambar diklasifikasikan ke dalam **tiga kategori**:
- 🐶 **Dog**
- 🐱 **Cat**
- 🐯 **Wild**

Model yang dikembangkan menggunakan pendekatan **transfer learning** dengan arsitektur **MobileNetV2**, karena sifatnya yang ringan namun tetap memberikan akurasi tinggi dibanding model lain seperti ResNet50 dan VGG16.

---

## 📂 Struktur Folder
animal_faces/
  
  ├── cat/
  
  ├── dog/
  
  └── wild/

Setiap folder berisi gambar wajah hewan berdasarkan kelas.

---

## 🛠️ Langkah-Langkah Proyek

### 1. 📦 Import Library
Menggunakan library:
- TensorFlow & Keras
- Matplotlib
- NumPy
- scikit-learn

### 2. 📁 Load Dataset
Dataset diekstrak dan dibagi ke dalam:
- 80% data latih
- 20% data validasi

Menggunakan `image_dataset_from_directory` dari TensorFlow.

### 3. 🔍 Visualisasi Dataset
Menampilkan contoh gambar dari setiap kelas untuk mengecek kualitas data dan distribusi kelas.
![Output](https://github.com/IchaAgni/Animal-Faces/blob/main/output.png)

### 4. 🔁 Augmentasi Data
Menggunakan teknik augmentasi citra dengan:
- Rotasi acak 10%
- Zoom acak 10%
- Translasi horizontal & vertikal 10%
- Flipping horizontal
  
```
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomFlip('horizontal')
])
```
### 5. 🧠 Arsitektur Model
Model dibangun dalam tiga tahap utama:
- Data Augmentation
- Preprocessing dan Transfer Learning
- Klasifikasi
  
```
  model = tf.keras.Sequential([
  data_augmentation,
    preprocess_input,
    base_model,  # MobileNetV2
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')  # Tiga kelas: cat, dog, wild
])
```
- Optimizer: Adam
- Loss Function: CategoricalCrossentropy
- Metric: Accuracy

### 6. 🧪 Pelatihan Model
Model dilatih selama 10 epoch dengan:
- EarlyStopping untuk menghentikan pelatihan jika validasi tidak meningkat.
- ModelCheckpoint untuk menyimpan model terbaik.
  
```
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
```

### 7. 📈 Evaluasi & Visualisasi Model
1. Akurasi Validasi Tertinggi: ~98%
2. Loss Validasi: Stabil dan menurun
 ![Accuracy & Loss](https://github.com/IchaAgni/Animal-Faces/blob/main/accuracy.png)
3. Confusion Matrix
![Confussion-Matrick](https://github.com/IchaAgni/Animal-Faces/blob/main/CM.png)
5. Classification Report
```text

              precision    recall  f1-score   support

           0     0.9890    0.9988    0.9938       808
           1     0.9908    0.9805    0.9857       771
           2     0.9884    0.9884    0.9884       779

    accuracy                         0.9894      2358
   macro avg     0.9894    0.9893    0.9893      2358
weighted avg     0.9894    0.9894    0.9894      2358
```

---

## ✅ Kesimpulan
Model klasifikasi wajah hewan dengan MobileNetV2 menunjukkan performa tinggi dengan akurasi validasi sekitar 98%. Proyek ini dapat dijadikan dasar sistem klasifikasi hewan berbasis gambar untuk aplikasi edukatif, pengenalan fauna, dan sebagainya.

## 👩‍💻 Kontributor
- Nisa Agni Afifah
- Submission Akhir - Fundamental Deep-Learnig (Dicoding)


