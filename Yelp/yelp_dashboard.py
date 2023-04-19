# Import required libraries
import pandas as pd
from io import BytesIO
import base64
from matplotlib import pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output



# Read the data into pandas dataframe
sentiment_df = pd.read_csv("Yelp/Data/Results.csv")
freq = pd.Series(" ".join(sentiment_df['review_nostop']).split()).value_counts().reset_index()
max_words = len(freq)

# Create a dash application
app = dash.Dash(__name__)

# Create an app layout
app.layout = html.Div(children=[html.H1('Sentiment Analysis of Yelp',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 40}),
                                dcc.Dropdown(id = 'site-dropdown',
                                             options=[{'label': 'Word Cloud', 'value': 'WordCloud'},
                                                      {'label': 'Word Frequency', 'value': 'WordFreq'},
                                                      {'label': 'Sentiment Distribution', 'value': 'WordDist'}],
                                             value='WordCloud',
                                             placeholder='Select mode of Analysis',
                                             searchable=True),
                                html.Br(),
                                html.P('Select maximum Amount of Words used'),
                                dcc.Slider(id='word-slider',
                                                min=0, 
                                                max=max_words,
                                                value=100),
                                html.Br(),
                                html.P('Select Sentiment Polarity Range'),
                                dcc.RangeSlider(id='sentiment-slider',
                                                min=0, 
                                                max=1, 
                                                value=[0, 1]),
                                html.Br(),
                                html.Div([html.Img(id='image_wc')], 
                                          style={'textAlign': 'center'})
                                ])


@app.callback(Output('image_wc', 'src'),
              Input(component_id='site-dropdown', component_property='value'),
              Input(component_id='word-slider', component_property='value'),
              Input(component_id='sentiment-slider', component_property='value'))

def make_word_cloud(entered_style ,selec_max_words, sentiment_range):
    if entered_style == 'WordCloud':
        data = sentiment_df[sentiment_df['polarity'] >= sentiment_range[0]]
        data = sentiment_df[sentiment_df['polarity'] <= sentiment_range[1]]
        freq = pd.Series(" ".join(data['review_nostop']).split()).value_counts().reset_index()
        d = {a: x for a, x in freq.values}
        wc = WordCloud(background_color="white",
                    max_words=selec_max_words,
                    max_font_size=40, 
                    relative_scaling=.5,
                    width=480,
                    height=360)
        wc.fit_words(d)
        image = wc.to_image()
        img = BytesIO()
        image.save(img, format='PNG')
        return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

    elif entered_style == 'WordDist':
        data = sentiment_df
        data['polarity_rounded'] = data['polarity'].round(1)
        z = sentiment_df['polarity_rounded'].value_counts()
        counts = pd.DataFrame(z).reset_index()
        counts.columns = ['polarity_rounded', 'Count_Column']
        plt.plot(list(range(100)))
        img = BytesIO()
        plt.savefig(img, format='PNG')
        return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())



# Run the app
if __name__ == '__main__':
    app.run_server()