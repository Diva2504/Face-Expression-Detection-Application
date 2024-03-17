import csv
import os
import shutil

# Tentukan path ke file CSV Anda dan direktori dataset output.
csv_files = ['train.csv', 'test.csv', 'dev.csv']  # Ganti dengan nama file CSV Anda.
image_directory = 'food-tfk-images'  # Ini adalah folder tempat gambar-gambar berada.
dataset_dir = 'dataset'  # Ini adalah tempat folder-label akan dibuat.

# Buat direktori dataset jika belum ada.
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Fungsi untuk membuat folder-label.
def create_label_folders(csv_file):
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Lewati baris header jika ada.
        for row in csv_reader:
            image_filename = row[0]  # Kolom pertama adalah nama file gambar.
            label = row[2]  # Kolom ketiga adalah nama label.

            # Buat folder untuk label jika belum ada.
            label_folder = os.path.join(dataset_dir, os.path.basename(csv_file)[:-4], label)
            if not os.path.exists(label_folder):
                os.makedirs(label_folder)

            # Salin atau pindahkan gambar ke folder-label yang sesuai.
            image_source_path = os.path.join(image_directory, image_filename)  # Perbarui path.
            image_dest_path = os.path.join(label_folder, image_filename)

            # Gunakan shutil.copy() untuk menyalin gambar atau shutil.move() untuk memindahkannya.
            shutil.copy(image_source_path, image_dest_path)

# Buat folder-label berdasarkan data dari setiap file CSV.
for csv_file in csv_files:
    create_label_folders(csv_file)
