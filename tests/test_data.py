import unittest
import pandas as pd
import numpy as np

class TestData(unittest.TestCase):

    def test_raw_train_not_manipulated(self):
        # perform a check to see that raw train data has not been manipulated

        # paths to training data csv files
        path_test = 'data/train.csv'
        path_raw  = '../data/raw/train.csv'

        # read data files using pandas
        data_test = pd.read_csv(path_test)
        data_raw  = pd.read_csv(path_raw)

        # turn data into a dataframe
        df_test = pd.DataFrame(data_test) 
        df_raw  = pd.DataFrame(data_raw) 

        # assert that data has not been altered
        self.assertTrue(df_test.equals(df_raw))

    def test_raw_test_data_manipulated(self):
        # perform a check to see that raw test data has not been manipulated

        # paths to testing data csv files
        path_test = 'data/test.csv'
        path_raw  = '../data/raw/test.csv'

        # read data files using pandas
        data_test = pd.read_csv(path_test)
        data_raw  = pd.read_csv(path_raw)

        # turn data into a dataframe
        df_test = pd.DataFrame(data_test) 
        df_raw  = pd.DataFrame(data_raw) 

        # assert that data has not been altered
        self.assertTrue(df_test.equals(df_raw))

    def test_raw_sample_data_manipulated(self):
        # perform a check to see that raw test data has not been manipulated

        # paths to testing data csv files
        path_test = 'data/sample_submission.csv'
        path_raw  = '../data/raw/sample_submission.csv'

        # read data files using pandas
        data_test = pd.read_csv(path_test)
        data_raw  = pd.read_csv(path_raw)

        # turn data into a dataframe
        df_test = pd.DataFrame(data_test) 
        df_raw  = pd.DataFrame(data_raw) 

        # assert that data has not been altered
        self.assertTrue(df_test.equals(df_raw))

    def test_clean_train_data_columns(self):
        # Ensure length of unique elements in each column is the same 
        # in clean and raw training data.

        # paths to raw and clean training data csv files
        path_raw    = '../data/raw/train.csv'
        path_clean  = '../data/clean/train.csv'

        # read the raw and clean train.csv files using pandas
        data_raw    = pd.read_csv(path_raw)
        data_clean  = pd.read_csv(path_clean)

        # turn raw and clean training data into a dataframe
        df_raw    = pd.DataFrame(data_raw) 
        df_clean  = pd.DataFrame(data_clean) 

        # Iterate over each column of training data and check that the
        # number of unique elements is the same in both clean and raw
        # training data
        for col in df_raw:
            unique_raw   = list(df_raw[col].unique())
            unique_clean = list(df_clean[col].unique())

            # Some one column in the raw training data contain a 0 and
            # nan values at the same time, so changin nan to zero by
            # cleaning the data reduces the number of unique elements of
            # that column because the nan's turn into 0's.
            # I take care of it by removing nan from the list of unique
            # elements in the raw column.
            # example: [0,1,2,nan] -> [0, 1, 2]
            if len(unique_raw) != len(unique_clean):
                unique_raw = [x for x in unique_raw if np.isnan(x) == False]
                print(f"Train data: Had to remove nan in unique values for column: {col}")
                # print(col)
                # print(len(unique_raw), len(unique_clean))
                # print(unique_raw)
                # print(unique_clean)
                # print(unique_raw)

            self.assertTrue(len(unique_raw) == len(unique_clean))

    def test_clean_test_data_columns(self):
        # Ensure length of unique elements in each column is the same 
        # in clean and raw testing data.

        # paths to raw and clean training data csv files
        path_raw    = '../data/raw/test.csv'
        path_clean  = '../data/clean/test.csv'

        # read the raw and clean test.csv files using pandas
        data_raw    = pd.read_csv(path_raw)
        data_clean  = pd.read_csv(path_clean)

        # turn raw and clean test data into a dataframe
        df_raw    = pd.DataFrame(data_raw) 
        df_clean  = pd.DataFrame(data_clean) 

        # Iterate over each column of testing data and check that the
        # number of unique elements is the same in both clean and raw
        # testing data
        for col in df_raw:
            unique_raw   = list(df_raw[col].unique())
            unique_clean = list(df_clean[col].unique())

            # Some one column in the raw testing data contain a 0 and
            # nan values at the same time, so changin nan to zero by
            # cleaning the data reduces the number of unique elements of
            # that column because the nan's turn into 0's.
            # I take care of it by removing nan from the list of unique
            # elements in the raw column.
            # example: [0,1,2,nan] -> [0, 1, 2]
            if len(unique_raw) != len(unique_clean):
                unique_raw = [x for x in unique_raw if np.isnan(x) == False]
                print(f"Test data: Had to remove nan in unique values for column: {col}")
                # print(col)
                # print(len(unique_raw), len(unique_clean))
                # print(unique_raw)
                # print(unique_clean)
                # print(unique_raw)

            self.assertTrue(len(unique_raw) == len(unique_clean))

    def test_data_type_of_clean_training_data(self):
        # Make sure that none of the columns in the clean training data
        # set contain data types of 'object' 

        # path to clean training data csv files
        path_clean  = '../data/clean/train.csv'

        # read the clean train.csv file using pandas
        data_clean  = pd.read_csv(path_clean)

        # turn clean train data into a dataframe
        df_clean  = pd.DataFrame(data_clean) 

        for col in df_clean:
            self.assertTrue(df_clean[col].dtypes != 'object')

    def test_data_type_of_clean_testing_data(self):
        # Make sure that none of the columns in the clean testing data
        # set contain data types of 'object' 

        # path to clean testing data csv files
        path_clean  = '../data/clean/test.csv'

        # read the clean test.csv file using pandas
        data_clean  = pd.read_csv(path_clean)

        # turn clean test data into a dataframe
        df_clean  = pd.DataFrame(data_clean) 

        for col in df_clean:
            self.assertTrue(df_clean[col].dtypes != 'object')


if __name__ == '__main__':
    unittest.main()
