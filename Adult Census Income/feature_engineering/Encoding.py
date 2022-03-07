from sklearn.preprocessing import LabelEncoder
from logger.Logging import log_class
from data_preprocessing.feature_classification import features
from data_preprocessing.Target_mapping import map
import os
import warnings
warnings.filterwarnings('ignore')

class encode:

    def __init__(self):
        self.folder='./Log_Files/'
        self.filename='encode.txt'
        self.df_obj=map()
        self.features=features()
        self.label=LabelEncoder()

        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)

        self.log_obj=log_class(self.folder,self.filename)

    def education_var_encode(self):

        """
                    Method: education_var_encode
                    Description: This method is used to encode Education variable in Dataset
                    Parameters: None
                    Return: Dataframe after encoding Education Variable

                    Version:1.0
        """

        try:
            dataframe=self.df_obj.out_var_mapping()
            dictonary = {' Bachelors': 13, ' HS-grad': 9, ' 11th': 7, ' Masters': 14, ' 9th': 5,
                     ' Some-college': 10, ' Assoc-acdm': 12, ' 7th-8th': 4, ' Doctorate': 16,
                     ' Assoc-voc': 11, ' Prof-school': 15, ' 5th-6th': 3, ' 10th': 6, ' Preschool': 1,
                     ' 12th': 8, ' 1st-4th': 2}

            dataframe['education']=dataframe['education'].map(dictonary)

            self.log_obj.log("INFO","Education variable encoded")

            for i in dataframe.columns:
                if i=='education-num':
                    dataframe.drop(columns='education-num',inplace=True)
                else:
                    pass

            self.log_obj.log("INFO","Education-num variable dropped")

            return dataframe

        except Exception as e:
            self.log_obj.log("ERROR","Error occured is"+'\t'+str(e))

    def encode_variables(self):

        """
                    Method: encode_variables
                    Description: This method is used to encode all categorical variables in Dataset
                    Parameters: None
                    Return: Dataframe after encoding all categorical variables

                    Version:1.0
        """

        try:
            self.log_obj.log("INFO","Encoding categorical variables")
            data=self.education_var_encode()
            categorical = self.features.categorical_features()

            for i in categorical:
                if i!='education':
                    data[i]=self.label.fit_transform(data[i].values.reshape(-1,1))
                else:
                    pass

            self.log_obj.log("INFO","Label encoding of variables is done")

            return data

        except Exception as e:
            self.log_obj.log("ERROR","Exception is :"+'\t'+str(e))



