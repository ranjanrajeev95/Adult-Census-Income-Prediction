import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from logger.Logging import log_class
import os

class loading_raw:

    def __init__(self):
        self.training_file='Dateset/adult.csv'
        self.folder='./Log_Files/'
        self.file_name="loading_raw_data.txt"

        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_obj=log_class(self.folder,self.file_name)


    def load_data(self):

        """
                    Method: load_data
                    Description: This method is used to load the dataset
                    Parameters: None
                    Return: DataFrame after loading the dataset

                    Version:1.0
        """

        try:
            self.log_obj.log("INFO","loading the dataset into pandas dataframe")
            self.dataframe=pd.read_csv(self.training_file)
            self.log_obj.log("INFO",'Raw data got loaded in database')

            return self.dataframe

        except Exception as e:
            self.log_obj.log("ERROR","Failed while loading dataset.Error is:"+str(e))


