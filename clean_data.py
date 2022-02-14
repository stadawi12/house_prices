# - clean_data.py

import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pyparsing
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
            print(f"Successfully saved file to {path_full}")

        else:
            print(f"Found filename {filename} already stored in 'data/clean'")
            print("If you want to override the file set override=True")
    else:
        # save data frame to a csv file in the data/clean directory
        df.to_csv(path_full)
        print(f"Successfully saved file to {path_full}")

# TODO: implement a correlate function that will correlate each column
# with the house_price column and keep only those that are above a certain
# threshold.
def correlate():
    pass

def plot_var(df, var: float, conditional: str ='<', n_plots: int = 4):
    """ Plots columns of data vs SalePrice, choice of columns 
    is based on variance of a column, we have to specify the 
    condition and a threshold. So for example, we can specify if we 
    want to plot all columns with variance greater than 10.
    
    Parameters
    ----------
    df : pandas.DataFrame
        This is the data frame we want to operate on and which we want
        to use to plot our graphs
    var : float 
        This is the threshold for our condition
    conditional : str (optional: default='<')
        Can either be ">" or "<", determines the conditional of the
        condition
    n_plots : int (optional: default=4)
        this is the number of plots we want to show, the plots will be
        laid out in a grid of size n_plots x n_plots.

    Returns
    -------
    plt.show() : matplotlib.method
        Plots the data we have filtered for

    """
    # Initialise fig and axis, constrained_layout eliminates overlap
    fig, ax = plt.subplots(n_plots, n_plots, constrained_layout=True)

    # set y_data (points on y-axis) to the SalePrice column
    y_data = df['SalePrice']

    # initialise counter and only increment when condition is true
    n = 0
    for col in df_train:

        # Determine which condition to use
        if conditional == "<":
            # Perform check
            condition = df[col].var() <= var
        else:
            # Perform check
            condition = df[col].var() >= var

        # Given condition is true...
        if condition:

            # set x_data to the column for which the condition was true
            x_data = df[col]

            # print diagnostic data
            print(n, n // n_plots, n % n_plots)

            # plot a scatter plot of column vs the SalePrice column
            ax[n//n_plots, n%n_plots].scatter(x_data, y_data)

            # set title of axis to the column name
            ax[n//n_plots, n%n_plots].set_title(col)

            # increment counter
            n += 1

            # if the counter is out of scope of our axis, break out of
            # the loop
            if n == n_plots*n_plots - 1:
                break
        
    # return the fig
    plt.show()

def plot_bar(df, n_unique: int = 4, n_plots: int = 4):
    """ This function plots the count of elements of a column as a bar
    chart. It constraints itself to columns with only n_unique elements
    or less.
    
    Parameters
    ----------
    df : pandas.DataFrame
        data frame that we will be working on
    n_unique : int (optional: default=4)
        largest number of unique elements of a column to consider
    n_plots : int (optional: default=4)
        number of plots to show at once, the layout is a grid of n_plots
        x n_plots axis.

    Returns
    -------
    plt.show() 
        returns the plotted data

    """
    # Initialise fig and axis, constrained_layout eliminates overlap
    fig, ax = plt.subplots(n_plots, n_plots, constrained_layout=True)

    # initialise counter
    n = 0

    # iterate over each column of a data frame
    for col in df:

        # obtain all unique elements of a column
        unique = df[col].unique()

        # only capture columns with unique elements <= n_unique
        if len(unique) <= n_unique:

            # Create a dictionary to keep count of each element in a
            # unique element of a column
            d = {c: 0 for c in unique}

            # turn elements of a column to a list
            col_list = df[col].tolist()

            # populate the dictionary by iterating over the column
            # elements and incrementing the correct value in the
            # dictionary by one
            for el in col_list: d[el] += 1

            # Extract the names and values of the dictionary after the
            # counting is complete
            names  = list(d.keys())
            values = list(d.values())

            # print diagnostic data
            print(n, n // n_plots, n % n_plots)

            # plot a bar chart of element counts in a column
            ax[n//n_plots, n%n_plots].bar(names, values)

            # set title of axis to the column name
            ax[n//n_plots, n%n_plots].set_title(col)

            # increment counter
            n += 1

            # if the counter is out of scope of our axis, break out of
            # the loop so we don't get an error
            if n == n_plots*n_plots :
                break

    # return the our figure
    plt.show()

if __name__ == '__main__':

    # set override bool
    override = False

    # specify path to data sets
    path_train = 'data/clean/train.csv'
    path_test  = 'data/raw/test.csv'

    # load data as suing pandas from a csv file
    df_train = pd.read_csv(path_train)
    df_test  = pd.read_csv(path_test)

    # # transform data to change integer values into integers (machine
    # # readable values)
    # df_train_transformed = transform(df_train)
    # df_test_transformed  = transform(df_test)

    # # save data to the clean directory inside data
    # save_data(df_train_transformed, 'train.csv', override=override)
    # save_data(df_test_transformed, 'test.csv', override=override)

    # plot_data(df_train)
    # plot_var(df_train, 0.01, conditional='<', n_plots=4)
    # plot_bar(df_train, n_unique=3)
