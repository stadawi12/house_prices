In this directory we have three stages of the data.

`raw`: this directory contains the raw, unprocessed data provided by kaggle.

`clean`: this directory contains data that has been transformed, all the string
values in the data have been transformed into integers and all nan
values have been set to "NaN" or 0, this is the clean data but has not
been processed, i.e. we keep all the columns of this data.

`train`: this directory contains data that is ready for training, processed data
where certain columns of the data have been removed due to a low
correlation score between their values and the 
column `house_price`.
