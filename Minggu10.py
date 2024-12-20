# Dataset
# Cuaca, Waktu, Suhu, Membawa_Payung
# Cerah, Pagi, Dingin, Tidak
# Cerah, Siang, Panas, Tidak
# Hujan, Pagi, Dingin, Ya
# Hujan, Sore, Sejuk, Ya
# Berawan, Pagi, Sejuk, Tidak
# Hujan, Siang, Panas, Ya

# Ini adalah dataset yang digunakan untuk melatih dan menguji model.
# Setiap entri berisi informasi tentang cuaca, waktu, suhu, dan apakah seseorang membawa payung.
data = [
    {"Cuaca": "Cerah", "Waktu": "Pagi", "Suhu": "Dingin", "Membawa_Payung": "Tidak"},
    {"Cuaca": "Cerah", "Waktu": "Siang", "Suhu": "Panas", "Membawa_Payung": "Tidak"},
    {"Cuaca": "Hujan", "Waktu": "Pagi", "Suhu": "Dingin", "Membawa_Payung": "Ya"},
    {"Cuaca": "Hujan", "Waktu": "Sore", "Suhu": "Sejuk", "Membawa_Payung": "Ya"},
    {"Cuaca": "Berawan", "Waktu": "Pagi", "Suhu": "Sejuk", "Membawa_Payung": "Tidak"},
    {"Cuaca": "Hujan", "Waktu": "Siang", "Suhu": "Panas", "Membawa_Payung": "Ya"},
]

# Fungsi untuk menghitung probabilitas tiap kelas
def calculate_probability(dataset, fitur, target):
    # Hitung probabilitas untuk tiap kelas
    probabilitas_kelas = {}
    total_data = len(dataset)
    for entry in dataset:
        label = entry[target]
        if label not in probabilitas_kelas:
            probabilitas_kelas[label] = 0
        probabilitas_kelas[label] += 1

    for label in probabilitas_kelas:
        probabilitas_kelas[label] /= total_data

    # Hitung probabilitas fitur terhadap kelas
    probabilitas_fitur = {}
    for entry in dataset:
        label = entry[target]
        for key, value in entry.items():
            if key == target:
                continue
            if (key, value, label) not in probabilitas_fitur:
                probabilitas_fitur[(key, value, label)] = 0
            probabilitas_fitur[(key, value, label)] += 1

    for key in probabilitas_fitur:
        fitur_count = sum(
            1 for entry in dataset if entry[target] == key[2]
        )
        probabilitas_fitur[key] /= fitur_count

    return probabilitas_kelas, probabilitas_fitur

# Fungsi untuk prediksi menggunakan Naive Bayes
def predict(input_data, probabilitas_kelas, probabilitas_fitur):
    hasil = {}
    for label in probabilitas_kelas:
        hasil[label] = probabilitas_kelas[label]
        for key, value in input_data.items():
            if (key, value, label) in probabilitas_fitur:
                hasil[label] *= probabilitas_fitur[(key, value, label)]
            else:
                hasil[label] *= 0  # Fitur tidak muncul pada kelas ini
    return max(hasil, key=hasil.get)

# Hitung probabilitas
prob_kelas, prob_fitur = calculate_probability(data, fitur=["Cuaca", "Waktu", "Suhu"], target="Membawa_Payung")

# Input data dari user
print("Masukkan data untuk prediksi:")
input_cuaca = input("Cuaca (Cerah/Hujan/Berawan): ")
input_waktu = input("Waktu (Pagi/Siang/Sore): ")
input_suhu = input("Suhu (Dingin/Panas/Sejuk): ")

# Data uji berdasarkan input user
test_data = {"Cuaca": input_cuaca, "Waktu": input_waktu, "Suhu": input_suhu}

# Prediksi hasil
prediction = predict(test_data, prob_kelas, prob_fitur)
# Mencetak hasil prediksi untuk data uji
print("Prediksi Membawa Payung:", prediction)
