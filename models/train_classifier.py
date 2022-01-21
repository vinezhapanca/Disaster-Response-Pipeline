import sys
from sqlalchemy import create_engine
import pandas as pd
import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle


def load_data(database_filepath):
    """
    The function to load the data from database. 
   
  
    Parameters:
    database_filepath = the location of database
  
  
    """
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('test', engine)
    X = df['message'].values
    Y = df.iloc[:, 4:].values
    category_names = df.iloc[:, 4:].columns
    return X,Y,category_names

def tokenize(text):
    """
    The transformations (lemmatization, convert to lower case, remove whitespaces) done to the text. 
   
  
    Parameters:
    text = the text to be transformed
  
    Returns:
    clean_tokens: the text which has been transformed
  
    """

    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    """
    The transformations (lemmatization, convert to lower case, remove whitespaces) done to the text. 
   
  
    Parameters:
    text = the text to be transformed
  
    Returns:
    clean_tokens: the text which has been transformed
  
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False)
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
        """
    The function to do model evaluation by fitting the model to test data to predict the classes, then compare the results with the actual classes.
   
  
    Parameters:
    X_test :  The dataset to be put into the model to be classified 
    Y_test :  The ground truth for the X_test. 
    category_names : The column names which predicted values need to be compared with Y_test
  
    Returns:
    printed model evaluation for each specified category_names
  
    """
    Y_pred= model.predict(X_test)
    Y_test = pd.DataFrame(Y_test)
    Y_pred = pd.DataFrame(Y_pred)
    Y_test.columns = category_names
    Y_pred.columns = category_names
    for i in category_names:
        print(classification_report(Y_test[i], Y_pred[i]))


def save_model(model, model_filepath):
     """
    Save the model into pickle format
   
  
    Parameters:
    model : the classifier model to be saved as pickle
    model_filepath : the destination location to put the pickle file
 
  
    """
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
