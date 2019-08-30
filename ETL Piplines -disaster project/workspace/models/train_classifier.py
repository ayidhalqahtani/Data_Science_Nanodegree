import pandas as pd 
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import nltk
from sklearn.pipeline import Pipeline
import sys

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC




def load_data(database_filepath,table_name='InsertTableName'):
    # load data from database

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql(table_name, con =engine)
    categories = df.columns[4:]

    X = df[['message']].values[:, 0]
    y = df[categories].values
    return X,y,categories



   


def tokenize(text):
    #Case Normalization & remove punctuation 
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) 
    
    #tokenization methods 
    
    words = word_tokenize(text)
    
    
    
    words = [w for w in words if w not in stopwords.words("english")]

    
    return words  

def build_model():

    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])
    
    parameters = {'vect__min_df': [1, 5],
              'tfidf__use_idf':[True, False],
              'clf__estimator__n_estimators':[10, 23], 
              'clf__estimator__min_samples_split':[2, 5, 9]}


    cv = GridSearchCV(pipeline, param_grid = parameters, verbose = 5)
    
    return cv




def Classification_report(actual, prediction,columns):
    for i in range(0, len(columns)):
        print(columns[i])
        print(" F1_score: {:.4f} % Accuracy: {:.4f}  % Precision: {:.4f} % Recall: {:.4f} ".format(
            f1_score(actual[:, i], prediction[:, i], average='weighted'),
            accuracy_score(actual[:, i], prediction[:, i]),
            precision_score(actual[:, i], prediction[:, i], average='weighted'),
            recall_score(actual[:, i], prediction[:, i], average='weighted')
        ))

def save_model(model, model_filepath):
    
    with open(model_filepath, 'wb') as f:
         pickle.dump(model, f)
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()