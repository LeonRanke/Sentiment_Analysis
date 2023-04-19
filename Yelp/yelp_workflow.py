# Import Dependencies
import re
import nltk
import time
import requests
import numpy as np
import pandas as pd

from textblob import Word
from textblob import TextBlob
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

# Class for getting reviews from Yelp
class Yelp_Reviews:
    def __init__(self, buisness):
        self.buisness = buisness
        self.url = f'https://www.yelp.com/biz/{self.buisness}?sort_by=date_desc'
        
    def get_reviews(self, page):
        reviews = []
        link = self.url + '?start=' + str(page*10)
        html = requests.get(link)
        if html.status_code == 200:
            soup = BeautifulSoup(html.text, 'html.parser')
            results = soup.find_all('span', {'lang': 'en'})
            for review in results:
                reviews.append(review.text)
            return reviews
        else:
            print('Page not found')
            return False

# Define a get reviews function
def get_reviews(buisness, num_pages):
    yelp = Yelp_Reviews(buisness)
    reviews = []
    for x in range(num_pages):
        print(f'Getting Page {x}')
        time.sleep(0.3)
        reviews.append(yelp.get_reviews(x))
    reviews_flattend = np.array(reviews).flatten()
    return reviews_flattend


# Preprocess Collected Reviews
def preprocess(reviews):
    df = pd.DataFrame(reviews, columns=['review'])
    stop_words = stopwords.words('english')

    # Lowercase
    df['review_lower'] = df['review'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    # Strip Punctuation
    df['review_nopunc'] = df['review_lower'].str.replace('[^\w\s]','', regex=True)
    # Remove Stopwords
    df['review_nostop'] = df['review_nopunc'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
    # Custom Stopwords list
    other_stopwords = ['one', 'get', 'go', 'im', '2', 'thru', 'tell','says', 'two']
    # Remove Custom Stopwords
    df['review_noother'] = df['review_nostop'].apply(lambda x: " ".join(x for x in x.split() if x not in other_stopwords))
    # Lemmatize
    df['cleaned_review'] = df['review_noother'].apply(lambda x: " ".join(Word(word).lemmatize() for word in x.split()))

    return df


# Calculate sentiment
def calculate_sentiment(df):
    df['polarity'] = df['cleaned_review'].apply(lambda x: TextBlob(x).sentiment[0])
    df['subjectivity'] = df['cleaned_review'].apply(lambda x: TextBlob(x).sentiment[1])
    # Return the Final Dataframe
    return df 



if __name__ == "__main__":
    buisness = input('Enter the yelp buisness name to be scraped: ')
    num_pages = input('Enter the number of pages to be scraped: ')
    reviews = get_reviews(buisness, int(num_pages))
    df = preprocess(reviews)
    sentiment_df = calculate_sentiment(df)
    sentiment_df.to_csv('Yelp/Data/Results.csv')