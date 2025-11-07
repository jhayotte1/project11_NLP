import nltk
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer


tok = RegexpTokenizer(r"[A-Za-z]+(?:'[A-Za-z]+)?")

def tokenize_text(txt):
    tokens = tok.tokenize(txt)
    tokens = [t.lower() for t in tokens]
    return tokens

def tot_tokens_and_vocab_size(df: pd.DataFrame):
    df["tokens"] = df.iloc[:, 4].apply(tokenize_text)
    total_tokens = df["tokens"].apply(len).sum()
    vocab = set(token for tokens in df["tokens"] for token in tokens)
    vocab_size = len(vocab)
    return (total_tokens, vocab_size)

def number_ebooks(df: pd.DataFrame):
    return df.shape[0]

def info_df(df: pd.DataFrame, name: str):
    tok, vocab = tot_tokens_and_vocab_size(df)
    number = number_ebooks(df)
    return({
        "Name": name,
        "Number of tokens": tok,
        "Vocabulary size": vocab,
        "Number of ebooks": number
    })

if __name__ == "__main__":
    hate_df = pd.read_csv("data/csv/hate.csv", sep=",", encoding="utf-8")
    love_df = pd.read_csv("data/csv/love.csv", sep=",", encoding="utf-8")
    summary = pd.DataFrame([
        info_df(love_df, "Love DataFrame"),
        info_df(hate_df, "Hate DataFrame")
    ])
    print(summary)
    cell_text = summary.astype(str).values.tolist()
    col_lables = list(summary.columns)
    n_rows = len(cell_text)
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.6 * max(n_rows, 1)))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=col_lables,
        cellLoc="center",
        colLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.3)
    table.auto_set_column_width(col=list(range(len(col_lables))))
    plt.title("DataFrames description", fontsize=14, fontweight="bold")
    plt.savefig(os.path.join("img", "dataframe_description_table.png"))
    plt.show()