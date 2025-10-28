import os
from pandas import read_csv
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

CSV_DIR = "data/csv"

if __name__ == "__main__":
    love = read_csv(os.path.join(CSV_DIR, "love.csv"))
    hate = read_csv(os.path.join(CSV_DIR, "hate.csv"))
    DATE_FORMAT = "%B %d, %Y"
    love_dates,hate_dates = love.iloc[:,3],hate.iloc[:,3]

    love_years = [datetime.strptime(d,DATE_FORMAT).year for d in love_dates]
    hate_years = [datetime.strptime(d,DATE_FORMAT).year for d in hate_dates]

    combined = pd.DataFrame({
        'Year': love_years + hate_years,
        'Category': ['Love']*len(love_years) + ['Hate']*len(hate_years)
    })

    sns.histplot(data=combined, discrete=True,x='Year', hue='Category')
    plt.xlabel("Years")
    plt.ylabel("Number of poems")
    plt.title("Publication date of poems categorized by Hate or Love")
    plt.savefig(os.path.join("img", "histogram_of_publication.png"))
        


