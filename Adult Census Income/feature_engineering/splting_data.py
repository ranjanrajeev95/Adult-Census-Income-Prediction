from logger.Logging import log_class
from feature_engineering.Encoding import encode
import os
import warnings
warnings.filterwarnings('ignore')

class split:

    def __init__(self):
        self.folder='./Log_Files/'
        self.filename='splitting_data.txt'
        self.df_obj=encode()


        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_obj=log_class(self.folder,self.filename)


    def splitting(self):

        """
                    Method: splitting
                    Description: This method is used to split the data into dependent and Independent variables
                    Parameters: None
                    Return: Independent and dependent variables

                    Version:1.0
        """

        try:
            self.log_obj.log("INFO","Splitting the data")
            data=self.df_obj.encode_variables()
            x=data.drop('salary',axis=1)
            y=data['salary']

            self.log_obj.log("INFO","splitting has been done")

            return x,y

        except Exception as e:
            self.log_obj.log("INFO","Exception occured is:"+'\t'+str(e))


