import unittest
import pandas as pd

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

        # paths to clean training data csv file
        path_raw  = '../data/clean/train.csv'

        # read the clean train.csv file using pandas
        data_raw  = pd.read_csv(path_raw)

        # turn clean training data into a dataframe
        df_raw  = pd.DataFrame(data_raw) 

    def test_clean_test_data_columns(self):
        pass


if __name__ == '__main__':
    unittest.main()
