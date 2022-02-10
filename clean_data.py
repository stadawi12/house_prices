# - clean_data.py

import pandas as pd
from sklearn import preprocessing
import os

"""
In this module we will aim to clean our data.
The house price data is presented in 81 columns, each column not
only has a different data type (data types of columns are float64, int64
or object), but also contains nan values, these are values of data type
np.nan which are hard to deal with, we will turn these into a string
"NaN" or the integer 0 depending on the data type of the column they are
found in.

First, we will deal with the nan values in our data. We will iterate over
each column of our data, determine the type of its values and if they
are of type object, we will find all the nan values of that column 
and turn them into a string "NaN" using the pd.DataFrame.fillna("NaN")
method. If the data type of a column is anything else, then we shall
use the same method but change its argument from "NaN" to 0.

As mentioned, some columns are of data type object which in our case 
means string, in order for our neural network to recognise those data 
entries we will need to turn them into integers (numbers). This is 
because it is easier to work with numbers than strings when it comes 
to machine learning.

To turn the values of the columns from strings to unique integers we
will use the sklearn library, specifically a method from preprocessing
module called LabelEncoder(), this function turns unique string objects
into corresponding integer values, for example, a list of strings:
    ['cat', 'dog', 'cat', 'zebra'] would turn into
    [1,2,1,3], 
you can see that the LabelEncoder() assigns the string 'cat' to an
integer value 1, 'dog' to 2 and 'zebra' to 3.
"""

def transform(df):

    # Generate a new copy of training data
    df_transformed = pd.DataFrame(columns=df.columns)

    # Initialise LabelEncoder from sklearn library
    le = preprocessing.LabelEncoder()

    # Turn nan's into strings "NaN" or 0's
    # Depending if that column data types are 'object' or not we will
    # turn nan's into "NaN" or 0, respectively 
    for col in df:

        # if column data type is an 'object', meaning string
        if df[col].dtypes == 'object':

            # Fill all nan values with the string "NaN"
            df[col].fillna("NaN", inplace=True)

        # If not object, i.e. int64 or float 64
        else:

            # Fill all nan values with 0's
            df[col].fillna(0, inplace=True)

    # Use LabelEncoder to turn string values into integers
    # For each column in training data
    for col in df:

        # if the data type of column is an object, transform to integers
        if df[col].dtypes == 'object':
            # Fit label encoder to unique string values of a column
            le.fit(df[col].unique())

            # use sklearn library to transform 
            transformed_values = le.transform(df[col])
            df_transformed[col] = transformed_values
        else:
            df_transformed[col] = df[col]

    return df_transformed

def save_data(df, filename, override=False):
    """
    This function will aim to save the transformed data (clean data)
    to the data/clean folder. If the clean data is already there we will
    not override it, by default, to override the clean data, set
    override to True.
    """
    # Path to data/clean directory
    path_dir = 'data/clean'

    # Make full path i.e. data/clean/train.csv
    path_full = os.path.join(path_dir, filename)

    # if the file exists do not override unless override=True
    if os.path.exists(path_full):

        if override:
            # save data frame to a csv file in the data/clean directory
            df.to_csv(path_full)

        else:
            print(f"Found filename {filename} already stored in 'data/clean'")
            print("If you want to override the file set override=True")
    else:
        # save data frame to a csv file in the data/clean directory
        df.to_csv(path_full)


if __name__ == '__main__':

    path_train = 'data/raw/train.csv'
    data_train = pd.read_csv(path_train)
    df_train   = pd.DataFrame(data_train)
    df_train_transformed = transform(df_train)
    save_data(df_train_transformed, 'train.csv')

