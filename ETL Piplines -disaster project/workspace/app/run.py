import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data

engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('InsertTableName', engine)

# load model
model = joblib.load("../models/mlmodel.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    releated_message = df[df['related']==1].groupby('genre').count()['message']
    not_releated = df[df['related']==0].groupby('genre').count()['message']
    gnames = list(releated_message.index)
    
    # Calculate proportion of each category with label = 1
    categories = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()/len(df)
    categories = categories.sort_values(ascending = False)
    categories_names = list(categories.index)
    
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=gnames,
                    y=releated_message,
                    name = 'Related'
                ),
                
                Bar(
                    x=gnames,
                    y=not_releated,
                    name = 'Not Related Messages'
                )
            ],

            'layout': {
                'title': 'Distribution of Messages  Genre and Related Status',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode': 'group'
            }
        },
        {
            'data': [
                Bar(
                    x=categories_names,
                    y=categories
                )
            ],

            'layout': {
                'title': 'Proportion of Messages by Category',
                'yaxis': {
                    'title': "Proportion"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -45
                }
            }
        },
        
          {  'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
