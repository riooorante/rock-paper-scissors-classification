# Technical Report: Klasifikasi Rock Scissors Paper

Proyek ini adalah model klasifikasi sederhana untuk permainan Rock, Scissors, Paper menggunakan berbagai arsitektur deep learning. Model yang digunakan termasuk AlexNet, VGG16, ResNet18, dan ResNet50. Proyek ini juga mencakup aplikasi web Flask untuk menguji model yang telah dilatih melalui antarmuka unggah gambar.

## Deskripsi Proyek

Tujuan dari proyek ini adalah untuk mengklasifikasikan gambar dari Rock, Scissors, dan Paper. Proyek ini terdiri dari:

- Melatih model menggunakan berbagai arsitektur pre-trained dan melakukan fine-tuning untuk tugas klasifikasi.
- Aplikasi Flask untuk memprediksi kelas (Rock, Scissors, atau Paper) berdasarkan gambar yang diunggah.

### Model yang Digunakan:
- **AlexNet**
- **VGG16**
- **ResNet18**
- **ResNet50**

### Dataset
Dataset yang digunakan untuk pelatihan terdiri dari gambar yang dikategorikan ke dalam tiga kelas: Rock, Scissors, dan Paper.

## Menjalankan Pelatihan Model

### Langkah 1: Install Dependensi
Pastikan Anda telah menginstal pustaka berikut:
- torch
- torchvision
- pandas
- flask
- pillow

Anda dapat menginstalnya menggunakan perintah berikut:
```
pip install torch torchvision pandas flask pillow
```

### Langkah 2: Melatih Model
Untuk melatih model, jalankan skrip `train.py` dengan argumen yang sesuai:

```
python train.py --model <nama_model> --data_dir <path_ke_dataset> --epochs <jumlah_epoch> --batch_size <batch_size> --lr <learning_rate>
```

**Argumen:**
- `--model`: Nama model yang akan dilatih (`alexnet`, `vgg16`, `resnet18`, `resnet50`).
- `--data_dir`: Path ke direktori dataset, yang harus berisi subdirektori `train`, `val`, dan `test`.
- `--epochs`: Jumlah epoch pelatihan (default: 10).
- `--batch_size`: Ukuran batch untuk pelatihan (default: 32).
- `--lr`: Learning rate untuk pelatihan (default: 0.001).

Contoh:
```
python train.py --model alexnet --data_dir ./data --epochs 10 --batch_size 32 --lr 0.001
```

Ini akan melatih model, mengevaluasi pada set validasi, dan menyimpan bobot model yang telah di-fine-tune di direktori `result_train_model`.

### Langkah 3: Hasil Pelatihan Model
Proses pelatihan akan menghasilkan file CSV (`model_name_training_results.csv`) yang berisi loss pelatihan, loss validasi, dan akurasi validasi per epoch. Model yang telah di-fine-tune akan disimpan dengan nama `model_name_fine_tuned.pth`.

## Menjalankan Aplikasi Flask

### Langkah 1: Menyiapkan Aplikasi Flask
Untuk memulai aplikasi Flask, pastikan Anda telah melatih model dan menyimpan bobot model.

1. Pastikan model telah di-fine-tune dan file bobot model (misalnya `alexnet_fine_tuned.pth`) ada di direktori yang sesuai.
2. Siapkan aplikasi Flask dengan menempatkan file model yang telah di-fine-tune (misalnya `alexnet_fine_tuned.pth`) di direktori yang sesuai.

### Langkah 2: Menjalankan Aplikasi Flask
Anda dapat menjalankan aplikasi Flask dengan perintah berikut:

```
python app.py
```

Ini akan memulai aplikasi Flask di `http://127.0.0.1:5000/`. Anda dapat mengakses antarmuka web dan mengunggah gambar untuk memprediksi apakah gambar tersebut adalah Rock, Scissors, atau Paper.

### Langkah 3: Alur Kerja Aplikasi Flask
- Aplikasi menyediakan antarmuka untuk mengunggah gambar.
- Setelah gambar diunggah, aplikasi Flask akan:
  1. Memproses gambar.
  2. Mengirimkan gambar melalui model yang telah dilatih.
  3. Mengembalikan kelas yang diprediksi (Rock, Scissors, atau Paper).

## Contoh Penggunaan

### 1. Melatih Model
Jalankan skrip pelatihan untuk fine-tuning model:
```
python train.py --model alexnet --data_dir ./data --epochs 10 --batch_size 32 --lr 0.001
```

### 2. Menjalankan Aplikasi Flask
Setelah model dilatih, jalankan aplikasi Flask:
```
python app/app.py
```

### 3. Akses Aplikasi Web
Akses aplikasi di `http://127.0.0.1:5000/` melalui browser Anda, unggah gambar, dan aplikasi akan memprediksi kelas gambar tersebut.

### 4. Hasil Prediksi
Gambar yang diunggah akan diprediksi oleh model dan kelas yang diprediksi (Rock, Scissors, atau Paper) akan ditampilkan pada antarmuka web.
