import os
import requests
import fasttext
import fasttext.util
import pandas as pd
import numpy as np
import gzip, shutil
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt 


MODEL_BIN = "cc.en.300.bin"

DATA_PATH = "data/csv"


def title_to_vector(title):
    return ft.get_sentence_vector(title)

def df_vector_to_matrix(df):
    return np.vstack(df["vectors"].values).astype(np.float32)

def matrix_statistics(sim_matrix):
    n = sim_matrix.shape[0]
    s = np.clip(sim_matrix, -1.0, 1.0)
    mask = np.triu(np.ones_like(s, dtype=bool), k=1)
    values = s[mask]
    mean_sim = np.mean(values)
    std_sim = np.std(values)
    max = np.max(values)
    min = np.min(values)
    return mean_sim, std_sim, max, min

def get_same_title(sim_matrix, df, threshold=0.999):
    n = sim_matrix.shape[0]
    mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
    pairs = np.argwhere((sim_matrix >= threshold) & mask)
    result = []
    for i, j in pairs:
        result.append({
            "title1": df.iloc[i, 1],
            "text_number1": df.iloc[i, 0],
            "author1": df.iloc[i, 2],
            "title2": df.iloc[j, 1],
            "text_number2": df.iloc[j, 0],
            "author2": df.iloc[j, 2],
            "similarity": sim_matrix[i, j]
        })
    return pd.DataFrame(result)

def plot_table_summary(summary):
    shown = summary.copy()
    num_cols = shown.select_dtypes(include=[float, int]).columns
    shown[num_cols] = shown[num_cols].round(9)

    cell_text = shown.astype(str).values.tolist()
    col_labels = list(shown.columns)
    n_rows = len(cell_text)
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.6 * max(n_rows, 1)))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.3)
    table.auto_set_column_width(col=list(range(len(col_labels))))
    plt.title("Title similarity for each Dataframe", fontsize=14, fontweight="bold")
    plt.savefig(os.path.join("img", "dataframes_title_similarity.png"))
    plt.show()


if __name__ == "__main__":
    if not os.path.exists(MODEL_BIN):
        print("Downloading FastText model...")
        fasttext.util.download_model('en', if_exists='ignore')
        print("Download complete.")
    else:
        print("FastText model already exists.")

    ft = fasttext.load_model(MODEL_BIN)

    df_love = pd.read_csv(os.path.join(DATA_PATH, "love.csv"), usecols=[0,1,2])
    df_hate = pd.read_csv(os.path.join(DATA_PATH, "hate.csv"), usecols=[0,1,2])

    df_love["vectors"] = df_love.iloc[:, 1].apply(title_to_vector)
    df_hate["vectors"] = df_hate.iloc[:, 1].apply(title_to_vector)

    love_matrix = df_vector_to_matrix(df_love)
    hate_matrix = df_vector_to_matrix(df_hate)

    love_sim = cosine_similarity(love_matrix)
    hate_sim = cosine_similarity(hate_matrix)

    love_mean_sim, love_std_sim, love_max_sim, love_min_sim = matrix_statistics(love_sim)
    hate_mean_sim, hate_std_sim, hate_max_sim, hate_min_sim = matrix_statistics(hate_sim)
    print(f"Love Titles Similarity - Mean: {love_mean_sim}, Std: {love_std_sim}, Max: {love_max_sim}, Min: {love_min_sim}")
    print(f"Hate Titles Similarity - Mean: {hate_mean_sim}, Std: {hate_std_sim}, Max: {hate_max_sim}, Min: {hate_min_sim}")

    summary = pd.DataFrame([
        {
            "mean_similarity": love_mean_sim,
            "similarity_standard_deviation": love_std_sim,
            "max_similarity": love_max_sim,
            "min_similarity": love_min_sim
        },
        {
            "mean_similarity": hate_mean_sim,
            "similarity_standard_deviation": hate_std_sim,
            "max_similarity": hate_max_sim,
            "min_similarity": hate_min_sim
        }
    ])

    plot_table_summary(summary)


    ##To get the duplicates names
    #love_duplicates = get_same_title(love_sim, df_love)
    #hate_duplicates = get_same_title(hate_sim, df_hate)

    #print(f"Number of duplicate love titles: {len(love_duplicates)}")
    #print(f"Number of duplicate hate titles: {len(hate_duplicates)}")
    #print("Duplicate Love Titles:")
    #print(love_duplicates)

    #print("Duplicate Hate Titles:")
    #print(hate_duplicates)