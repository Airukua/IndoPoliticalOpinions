import pandas as pd

def get_data(data):
    df = pd.read_csv(data)
    df['author_anon'] = pd.factorize(df['author'])[0] + 1
    df['author_anon'] = df['author_anon'].apply(lambda x: f"user_{x}")
    df['author'] = df['author_anon']
    df = df.drop(columns='author_anon')
    df.to_csv(data, index=False)

dataset = ['scrapper/debat_tv_one.csv',
        'scrapper/Ijasah_ditunjukan.csv',
        'scrapper/komentar_pihak_yang_dituduh.csv',
        'scrapper/komentar_tokoh_politik.csv',
        'scrapper/teman_kuliah.csv'
        ]

if __name__ == "__main__":
    for data in dataset:
        get_data(data)
    print("[INFO] Anonymization completed")
