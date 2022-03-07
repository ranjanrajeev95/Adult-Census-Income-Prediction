import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from data_preprocessing.preprocessing import Data_transform
from logger.Logging import log_class
import os
import pymongo
import json

class Datadb:

    def __init__(self,dbname,collection):
        self.client = pymongo.MongoClient('mongodb://localhost:27017')
        self.dbname=self.client[dbname]
        self.collection=self.dbname[collection]
        self.obj_trans=Data_transform()

        self.folder='./Log_Files/'
        self.file_name='database.txt'

        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)

        self.log_obj=log_class(self.folder,self.file_name)


    def to_csv(self):

        """
                        Method:  to_csv
                        Description: This method is used to save data from database into csv format
                        Parameters: None
                        Return: CSV file from database

                        Version:1.0
        """

        try:
            results=self.collection.find()
            df=pd.DataFrame(results).drop(columns="_id")

            if not os.path.isdir('./Data_from_database/'):
                os.mkdir('./Data_from_database/')

            df.to_csv('./Data_from_database/'+'Dataset.csv',index=False)
            self.log_obj.log("INFO","Training data saved to csv file")

            return df

        except Exception as e:
            self.log_obj.log("ERROR","Exception is :"+'\t'+str(e))



    def insert_data(self):

        """
                        Method:  insert_data
                        Description: This method is used to insert data into database
                        Parameters: None
                        Return: None

                        Version:1.0
        """

        try:
            df=self.obj_trans.remove_duplicates()
            records=list(json.loads(df.T.to_json()).values())
            self.collection.insert_many(records)
            self.to_csv()
            self.log_obj.log("INFO","Data inserted into Database")

        except Exception as e:
            self.log_obj.log("ERROR","Exception occured in Insertion of records in DB.Exception is:"+'\t'+str(e))







