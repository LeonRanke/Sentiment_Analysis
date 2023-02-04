# Import Dependencies
import re
import nltk
import requests
import numpy as np
import pandas as pd

from textblob import Word
from textblob import TextBlob
from bs4 import BeautifulSoup
from nltk.corpus import stopwords




# Define a get reviews function
def get_reviews():
    links = [f'https://www.yelp.com/biz/mcdonalds-los-angeles-106?start={10+10*x}' for x in range(12)]
    links.insert(0, 'https://www.yelp.com/biz/mcdonalds-los-angeles-106')
    regex = re.compile('raw__')

    reviews = []
    for link in links:
        html = requests.get(link)
        if html.status_code == 200:
            soup = BeautifulSoup(html.text, 'html.parser')
            results = soup.find_all('span', {'lang': 'en'}, class_=regex)
            for review in results:
                reviews.append(review.text)
        else:
            print('[HTML ERROR] Did not recive HTML Status code 200 (ok) but ', html.status_code)
    
    return reviews


# Preprocess Collected Reviews
def preprocess(reviews):
    df = pd.DataFrame(np.array(reviews), columns=['review'])
    stop_words = stopwords.words('english')

    # Lowercase
    df['review_lower'] = df['review'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    # Strip Punctuation
    df['review_nopunc'] = df['review_lower'].str.replace('[^\w\s]','')
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
    Link = input('Enter the link to be scraped: ')
    print(Link)
    reviews = get_reviews()
    df = preprocess(reviews)
    sentiment_df = calculate_sentiment(df)
    sentiment_df.to_csv('Data/Results.csv')