import pandas as pd

def build_global_author_mapping(file_paths):
    """
    Mengumpulkan semua author unik dari daftar file,
    lalu membuat mapping global ke user_1, user_2, dst.
    """
    all_authors = set()
    for path in file_paths:
        df = pd.read_csv(path)
        all_authors.update(df['author'].dropna().unique())
    
    author_map = {author: f"user_{i+1}" for i, author in enumerate(sorted(all_authors))}
    return author_map

def apply_anonymization(file_path, author_map):
    """
    Membaca file CSV, mengganti kolom 'author' dengan nilai anonim,
    lalu menyimpan kembali ke file yang sama.
    """
    df = pd.read_csv(file_path)
    df['author'] = df['author'].map(author_map)
    df.to_csv(file_path, index=False)

def anonymize_authors_globally(file_paths):
    """
    Fungsi utama yang menjalankan seluruh proses anonimisasi global.
    """
    author_map = build_global_author_mapping(file_paths)
    for path in file_paths:
        apply_anonymization(path, author_map)
    return author_map 
