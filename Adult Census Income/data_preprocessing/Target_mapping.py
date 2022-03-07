from logger.Logging import log_class
from data_preprocessing.feature_classification import features
from data_preprocessing.preprocessing import Data_transform
import os
import warnings
warnings.filterwarnings('ignore')

class map:

    def __init__(self):
        self.obj_pre=Data_transform()
        self.ob_feature=features()
        self.folder='./Log_Files/'
        self.filename='target_mapping.txt'

        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_obj=log_class(self.folder,self.filename)

    def out_var_mapping(self):

        """
                    Method: out_var_mapping
                    Description: This method is used to map output variables
                    Parameters: None
                    Return: DataFrame after mapping output variable

                    Version:1.0
        """

        try:
            data=self.obj_pre.remove_duplicates()
            data['salary']=data['salary'].map({' <=50K':0,' >50K':1})

            self.log_obj.log("INFO","Target variable mapping is done")

            return data

        except Exception as e:
            self.log_obj.log("ERROR","Exception occured is :"+'\t'+str(e))






