# Adult_Income_census:
## Problem Statement:
<p>The prominent inequality of wealth and income is a huge concern especially in the United States. The likelihood of diminishing poverty is one valid reason to reduce the world's surging level of economic inequality. The principle of universal moral equality ensures sustainable development and improve the economic stability of a nation. Governments in different countries have been trying their best to address this problem and provide an optimal solution. This study aims to show the usage of machine learning techniques in providing a solution to the income equality problem. The UCI Adult Dataset has been used for the purpose. Classification has been done to predict whether a person's yearly income in US falls in the income category of either greater than 50K Dollars or less equal to 50KDollars category based on a certain set of attributes.<p> 
  
  ## Approach
  <p> The main goal is to predict the income of a person is ess than or equal to 50k or more than 50k.<p>
  <pre>
  <li> Data Collection                    : We have collecetd data from kaggle platform
  <li> Log creation                       : We try create logging in order keep track and to understand workflow
  <li> Data insertion into Database       : We try insert data into Database MangoDb
  <li> Exploratory data analysis          : We used Pandas profiling to for EDA part we generated a 
                                            report and stored in analysis folder
  <li> Feature Engineering                : We try fill missing values,encode variables and 
                                            standardize the variables
  <li> Model creation and Model tunning   : We try create Xgboost classifier and Random forest classfier and by 
                                            using grid search cv and cross validation we try to tune the model
 <li> Pickel file creation                : After Creation of model we try to save model in pickle format .
 <li>Frontend and Deployment              : We used HTML for frontend and Deployed or model in Heroku platform 
 
 
 ## Deployment Links (Heroku)
link : https://adult-income-census.herokuapp.com/ <br>
 
 ## WorkFlow
 ![alt-text](https://github.com/prakash0007/Adult_Income_census/blob/main/Images/Workflow.png)
 
# Snippets of Project
#### 1) Training:

![alt-text](https://github.com/prakash0007/Adult_Income_census/blob/main/Images/Training_image.png)

#### 2) Home Page:

![alt-text](https://github.com/prakash0007/Adult_Income_census/blob/main/Images/index_page.png)

#### 3) Result Prediction page:

![alt-text](https://github.com/prakash0007/Adult_Income_census/blob/main/Images/output_text.png)

## Technologies Used
<pre> 
1. Python 
2. Sklearn
3. Flask
4. Html
5. Pandas, Numpy 
7. Database MangoDb 
8.Heroku colud
</pre>

## HLD , LLD & WireFrame
link : https://drive.google.com/drive/folders/1R7lT6uUGeV1WDm5FMoDUHteeaN0CGJT6?usp=sharing

## Help Me Improve
<p> Hello Reader if you find any bug please consider raising issue I will address them asap.</p>





   
