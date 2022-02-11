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


if __name__ == '__main__':
    unittest.main()
