import warnings
warnings.filterwarnings('ignore')
import numpy as np
from logger.Logging import log_class
from data_preprocessing.loading_raw_data import loading_raw
from data_preprocessing.feature_classification import features
import os

class validation:

    def __init__(self):
        self.folder='./Log_Files/'
        self.filename='raw_validation.txt'
        self.df_object = loading_raw()
        self.features = features()

        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_obj=log_class(self.folder,self.filename)

    def data_stdzero(self):

        """
                    Method: data_stdzero
                    Description: This method is used to check if standard deviation is Zero in the dataset
                    Parameters: None
                    Return: DataFrame after removing zero standard deviation columns

                    Version:1.0
        """

        try:
            self.log_obj.log("INFO","Checking if standard deviation is zero or not for numerical features ")
            dataframe=self.df_object.load_data()
            numerical_features=self.features.numerical_features()

            for feature in numerical_features:
                if dataframe[feature].std()==0:
                    dataframe.drop(columns=feature,axis=1,inplace=True)
            self.log_obj.log("INFO","Zero std deviation columns are removed")
            return dataframe

        except Exception as e:
            self.log_obj.log("ERROR","Occured Exception is :"+"\t"+str(e))


    def data_whole_missing(self):

        """
                    Method: data_whole_missing
                    Description: This method is used to check if entire column has missing values
                    Parameters: None
                    Return: DataFrame after removing columns with entire missing values

                    Version:1.0
        """

        try:
            self.log_obj.log("INFO","checking if there is column with whole missing values")
            data=self.data_stdzero()

            for i in data.columns:
                if data[i].isnull().sum()==len(data[i]):
                    data.drop(columns=i,axis=1,inplace=True)
                else:
                    pass

            self.log_obj.log("INFO","Data with whole missing value row is dropped")

            return data

        except Exception as e:
            self.log_obj.log("ERROR","Exception is:"+'\t'+str(e))

    def finding_null(self):
        """
                    Method: finding_null
                    Description: This method is used to check if their are any null values and replacing them with nan
                    Parameters: None
                    Return: DataFrame after checking null values

                           Version:1.0
         """


        try:
            self.log_obj.log("INFO","Replacing null type values with NAN value")
            dataFrame=self.data_whole_missing()
            categorical=self.features.categorical_features()
            numerical=self.features.numerical_features()

            for feature in numerical:
                dataFrame[feature]=dataFrame[feature].replace(' ?',np.nan)

            for feature in categorical:
                dataFrame[feature]=dataFrame[feature].replace(' ?',np.nan)

            self.log_obj.log("INFO","Replaced null type values with NAN values")
            return dataFrame

        except Exception as e:
            self.log_obj.log("ERROR","Exception is:"+'\t'+str(e))





