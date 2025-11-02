# project11_NLP
Project 11 : Natural Language Processing : Love and Hate in Poetry

# Prerequisites

## Gutenberg Index

[Machine readable index](https://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2)

We have used this index for scraping and identifying similar poems, the file hierarchy should look like this after unzipping:
```
project11_NLP/
├── gutenberg_rdf/
│   └── cache/
│       └── epub/
│           └── (RDF FILES)
```

## Dependencies

Install the required dependencies:
```bash
pip install -r requirements.txt
```

# Tasks

## 1

Task is implemented in 'dataframe_stat.py'

![DataFrame description table (Number of token, vocabulary size, number of books)](./img/dataframe_description_table.png)

## 2

Task is implemented in `histogram_of_publication.py`:

![Histogram of publication dates based on category](./img/histogram_of_publication.png)

## 3

![Love and Hate words quantified](./img/love_and_hate_words.png)

## 4

![Love in Love dataframe context word wordcloud](./img/love_in_love_context_wordcloud.png)
![Love in Hate dataframe context word wordcloud](./img/love_in_hate_context_wordcloud.png)
![Hate in Love dataframe context word wordcloud](./img/hate_in_love_context_wordcloud.png)
![Hate in Hate dataframe context word wordcloud](./img/hate_in_hate_context_wordcloud.png)

## 5


## 6


## 7 and 8

Average Semantic Similarity between top 100 frequent words and vocabularies:
```
Love & Love Vocab: 0.35974183678627014
Hate & Love Vocab: 0.3619268536567688
Hate & Hate Vocab: 0.3463706374168396
Love & Hate Vocab: 0.3427005112171173
``` 


## 9


## 10


## 11
