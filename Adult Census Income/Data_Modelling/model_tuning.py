from logger.Logging import log_class
from feature_engineering.scaling_tranform import Data_scaling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,accuracy_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class Model_Finder:

    def __init__(self):
        self.folder='./Log_Files/'
        self.filename='model_tuning.txt'
        self.df_obj=Data_scaling()
        self.model_list=[]
        self.dict_model=[]
        self.rfc=RandomForestClassifier()
        self.xgb=XGBClassifier()

        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)

        self.log_obj=log_class(self.folder,self.filename)

    def get_best_params_for_random_forest(self):

        """
               Method:get_best_params_for_random_forest
               Description:This method is used to get best parameters from Random_forest_Classifier to the model
               Parameters:None
               Return:Parameters for Individual model

               Version:1.0
        """


        try:
            self.log_obj.log("INFO","Entered into best_params_class for random_forest classifier")
            x_train,x_test,y_train,y_test=self.df_obj.scaling()
            self.param_grid={'criterion': ['gini'], 'max_depth': [890], 'max_features': ['auto'],
                             'min_samples_leaf': [2, 4, 6], 'min_samples_split': [0, 1, 2, 3, 4],
                              'n_estimators': [600, 700, 800, 900, 1000]}
            self.grid=GridSearchCV(estimator=self.rfc,param_grid=self.param_grid,cv=10,n_jobs=-1,verbose=2)
            self.grid.fit(x_train,y_train)

            self.log_obj.log("INFO","Grid_Search cv is performed ")
            self.criterion=self.grid.best_params_['criterion']
            self.max_depth=self.grid.best_params_['max_depth']
            self.max_features=self.grid.best_params_['max_features']
            self.min_samples_leaf=self.grid.best_params_['min_samples_leaf']
            self.min_samples_split=self.grid.best_params_['min_samples_split']
            self.n_estimators=self.grid.best_params_['n_estimators']

            self.log_obj.log("INFO","Random forest modelling fitting has started")

            self.rfc=RandomForestClassifier(n_estimators=self.n_estimators,criterion=self.criterion,
                                            max_depth=self.max_depth,max_features=self.max_features,
                                            min_samples_leaf=self.min_samples_leaf,
                                            min_samples_split=self.min_samples_split)
            self.rfc.fit(x_train,y_train)

            self.log_obj.log("INFO",f"Best random forest parameters are {self.grid.best_params_}")
            self.log_obj.log("INFO","Random Forest Modelling completed")

            return self.rfc

        except Exception as e:
            self.log_obj.log("ERROR", "Exception occured at Random forest modelling, Exception is :"+'\n'+str(e))


    def get_best_params_for_xg_boost(self):

        """
               Method:get_best_params_xg_boost
               Description:This method is used to get best parameters from xg_boost_Classifier to the model
               Parameters:None
               Return:Parameters for Individual model

               Version:1.0
        """

        try:
            self.log_obj.log("INFO","Entered into best_params for xg boost class")
            x_train,x_test,y_train,y_test=self.df_obj.scaling()
            self.param_grid_xg={
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.5, 0.1, 0.01, 0.001],
                'max_depth': [10, 120, 230, 340]
            }

            self.grid_xg=GridSearchCV(XGBClassifier(objective='binary:logistic'),self.param_grid_xg,verbose=3,
                                   cv=5)
            self.grid_xg.fit(x_train,y_train)

            self.learning_rate=self.grid_xg.best_params_['learning_rate']
            self.n_estimators=self.grid_xg.best_params_['n_estimators']
            self.max_depth=self.grid_xg.best_params_['max_depth']

            self.log_obj.log('INFO',"Xg_boost modelling has started")
            self.xgb=XGBClassifier(learning_rate=self.learning_rate,n_estimators=self.n_estimators,
                                   max_depth=self.max_depth)

            self.xgb.fit(x_train,y_train)
            self.log_obj.log("INFO",f"Best xg_boost parameters are :{self.grid_xg.best_params_}")
            self.log_obj.log("INFO","XGboost modelling has been completed")

            return self.xgb

        except Exception as e:
            self.log_obj.log("INFO","Occured Exception is:" +'\n'+str(e))


    def get_best_model(self):

        """
                       Method:get_best_model
                       Description:This method is used to get best model from set from the models by comparing their scores
                       Parameters:None
                       Return: Best model name

                       Version:1.0
        """

        try:
            self.log_obj.log("INFO","Enter get_best_model method in Model_finder_class ")
            x_train,x_test,y_train,y_test=self.df_obj.scaling()
            self.xgboost=self.get_best_params_for_xg_boost()
            self.prediction_xgboost=self.xgboost.predict(x_test)

            if len(y_test.unique())==1:
                self.xgboost_score=accuracy_score(y_test,self.prediction_xgboost)
                self.log_obj.log("INFO",f"Accuracy of Xgboost model is {self.xgboost_score}")
            else:
                self.xgboost_score=roc_auc_score(y_test,self.prediction_xgboost)
                self.log_obj.log("INFO",f"Roc_Auc score is {self.xgboost_score}")


            self.random_forest=self.get_best_params_for_random_forest()
            self.prediction_random_forest=self.random_forest.predict(x_test)

            if len(y_test.unique())==1:
                self.random_forest_score=accuracy_score(y_test,self.prediction_random_forest)
                self.log_obj.log("INFO",f"Accuracy of random forest is {self.random_forest_score}")
            else:
                self.random_forest_score=roc_auc_score(y_test,self.prediction_random_forest)
                self.log_obj.log("INFO",f"Auc and Roc score of Random forest is {self.random_forest_score}")


            self.log_obj.log("INFO","Selection of best model ")


            if(self.random_forest_score<self.xgboost_score):
                self.best_model_name='xgboost'
                self.best_model=self.xgboost
                self.best_score=self.xgboost_score
            else:
                self.best_model_name='Random_forest'
                self.best_model=self.random_forest
                self.best_score=self.random_forest_score



            self.log_obj.log("INFO","Best model selection is done")
            self.log_obj.log("INFO","Saving model as pickle file")

            if not os.path.isdir('./bestmodel/'):
                os.mkdir('./bestmodel/')

            with open('./bestmodel/'+self.best_model_name+'.pkl','wb') as file:
                pickle.dump(self.best_model,file)

            self.log_obj.log("INFO",f"Best model saved in file ,Best model is:{self.best_model_name}")

            return self.best_model_name

        except Exception as e:
            self.log_obj.log("INFO","Exception occured at get_best_model  is:"+str(e))














