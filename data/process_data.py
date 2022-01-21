import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    The function to load the messages and categories data, then combine these tables based on mutual id.
 
  
    Parameters:
    messages_filepath : The location of disaster_messages.csv file
    categories_filepath : The location of disaster_categories.csv file
  
    Returns:
    df: The resulting table, which is formed by joining messages and categories dataframes based on id. 
  
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = "id")
    
    return df




def clean_data(df):
     """
    The function to do data preprocessing. 
    
    The procedures done here are:
    1. duplicate the categories column as a different dataframe. 
    2.split it into 36 columns (the number of different categories), with ; as the separator
    3.extract the categories names (without the number part) to be put as the header
    4.cleanse the entries to only take the number part and convert to binary (0 and 1)
    5. replace the categories column of the existing table with the new categories columns
 
  
    Parameters:
    df : the table to be cleansed
  
    Returns:
    df: The resulting table, which has been preprocessed 
  
    """
    categories = df['categories'].str.split(";",n=36, expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[0:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype(int)
    categories[categories==2] = 1
    df = df.drop('categories',1)
    df = pd.concat([df,categories], axis = 1)
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """
    The function to save the dataframe into database. 
   
  
    Parameters:
    df : the table to be put in database
    database_filename : the name of database
  
  
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('test',engine, index=False)
    
    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
