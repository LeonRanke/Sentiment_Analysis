import re
import time
import deepl
import requests
import numpy as np
import pandas as pd
import detectlanguage

from textblob import Word
from textblob import TextBlob
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

   
# Define a get reviews function
def get_reviews(link, num_pages):
    # Set up hedder to Scrape Amazon
    HEDDERS = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7", 
               "Accept-Encoding": "gzip, deflate", 
               "Dnt": "1", 
               "Sec-Gpc": "1", 
               "Upgrade-Insecure-Requests": "1", 
               "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36", 
               "X-Amzn-Trace-Id": "Root=1-643fd868-3ca2c6ce01ed0a2f34795b57"}

    # Construct list of links to scrape multiple pages
    links = []
    for page in range(num_pages):
        links.append(link + '&pageNumber=' + str(page))

    # Scrape all links in the constructed list
    reviews = []
    for link in links:
        html = requests.get(link, headers=HEDDERS)
        if html.status_code == 200:
            # HTML response was sucssesfull
            soup = BeautifulSoup(html.text, 'html.parser')
            results = soup.find_all('span', {'data-hook': 'review-body'})
            for review in results:
                reviews.append(review.text.replace('\n', ''))
        else:
            # HTML response was unsuccsessfull
            print('[BAD HTML RESPONSE] Response Code =', html.status_code)
    
    return reviews

    

# Define a translate reviews function
def translate_reviews(reviews):
    detectlanguage.configuration.api_key = ""
    translator = deepl.Translator("")

    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    
    filterd_reviews = []
    for review in reviews:
        temp = regrex_pattern.sub(r'', review)
        if temp == '':
            pass
        else:
            filterd_reviews.append(temp)

    # Detect languges of all reviews
    languages = []
    for review in filterd_reviews:
        language = detectlanguage.detect(review)
        languages.append(language[0]['language'])

    # Construct Data frame containing review and Language
    df = pd.DataFrame({'Review': filterd_reviews, 'Language': languages})

    # Translate reviews
    translations = []
    for row in df.iterrows():
        review = row[1][0]
        language = row[1][1]
        if language == 'en':
            translation = review
        elif language in ['ie', 'sr']:
            translation = 'language not supported'
        else:
            translation = translator.translate_text(review, target_lang='en-gb', source_lang=language)
            
        translations.append(translation)
        
    # Add Translation to dataframe
    df['Translated'] = translations
    df.to_csv('Amazon/Data/Reviews_Translated.csv')

# Preprocess Collected Reviews
def preprocess():
    df = pd.read_csv('Amazon/Data/Reviews_Translated.csv')
    stop_words = stopwords.words('english')

    #  Lowercase
    df['review_lower'] = df['Translated'].apply(lambda x: " ".join(x.lower() for x in x.split()))
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

    df.to_csv('Amazon/Data/Reviews_Preprocessed.csv')
    return df

# Calculate sentiment
def calculate_sentiment(df):
    df['polarity'] = df['cleaned_review'].apply(lambda x: TextBlob(x).sentiment[0])
    df['subjectivity'] = df['cleaned_review'].apply(lambda x: TextBlob(x).sentiment[1])
    # Return the Final Dataframe
    return df 

if __name__ == "__main__":
    link = input('Enter a Amazon asin code to be scraped: ')
    num_pages = input('Enter the number of pages to be scraped: ')
    reviews = get_reviews(link, int(num_pages))
    print(len(reviews))
    #translate_reviews(reviews)
    #df = preprocess()
    #sentiment_df = calculate_sentiment(df)
    #sentiment_df.to_csv('Amazon/Data/Results.csv')