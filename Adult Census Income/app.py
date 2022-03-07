from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import os
import pickle
from flask_cors import cross_origin



app = Flask(__name__)

"""
Loading pickle file
"""
file = os.listdir('./bestmodel/')[0]
model = pickle.load(open('./bestmodel/'+file, 'rb'))
scaler = pickle.load(open('standard_scalar.pkl', 'rb'))


@app.route('/',methods=['POST','GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':


        Age= request.form['Age']

        Work_Class=request.form['Work_Class']

        if(Work_Class=='State-gov'):
            Work_Class=6
        elif(Work_Class=='Self-emp-not-inc'):
            Work_Class=5
        elif(Work_Class=='Private'):
            Work_Class=3
        elif (Work_Class == 'Local-gov'):
            Work_Class= 0
        elif (Work_Class == 'Self-emp-inc'):
            Work_Class = 1
        elif (Work_Class == 'Without-pay'):
            Work_Class = 7
        else:
            Work_Class = 2


        Final_Weight=int(request.form['Final_Weight'])

        Education=request.form['Education']

        if(Education=='Bachelors'):
            Education=13
        elif(Education=='HS-grad'):
            Education=9
        elif(Education=='11th'):
            Education=7
        elif(Education=='Masters'):
            Education=14
        elif(Education=='9th'):
            Education=5
        elif(Education=='Some-college'):
            Education=10
        elif(Education=='Assoc-acdm'):
            Education=12
        elif(Education=='7th-8th'):
            Education=4
        elif(Education=='Doctorate'):
            Education=16
        elif(Education=='Assoc-voc'):
            Education=11
        elif(Education=='Prof-school'):
            Education=15
        elif(Education=='5th-6th'):
            Education=3
        elif(Education=='10th'):
            Education=6
        elif(Education=='Preschool'):
            Education=1
        elif(Education=='12th'):
            Education=8
        else:
            Education=2


        Marital_Status=request.form['Marital_Status']
        if(Marital_Status=='Never-married'):
            Marital_Status= 4
        elif (Marital_Status == 'Married-civ-spouse'):
            Marital_Status = 2
        elif (Marital_Status == 'Divorced'):
            Marital_Status = 0
        elif (Marital_Status == 'Married-spouse-absent'):
            Marital_Status = 3
        elif (Marital_Status == 'Separated'):
            Marital_Status = 5
        elif (Marital_Status == 'Married-AF-spouse'):
            Marital_Status = 1
        else:
            Marital_Status = 6



        Occupation=request.form['Occupation']
        if(Occupation=='Adm-clerical'):
            Occupation=0
        elif (Occupation == 'Exec-managerial'):
            Occupation = 3
        elif (Occupation == 'Handlers-cleaners'):
            Occupation = 5
        elif (Occupation == 'Prof-specialty'):
            Occupation = 9
        elif (Occupation =='Other-service'):
            Occupation = 7
        elif (Occupation == 'Sales'):
            Occupation = 11
        elif (Occupation == 'Craft-repair'):
            Occupation = 2
        elif (Occupation == 'Transport-moving'):
            Occupation = 13
        elif (Occupation == 'Farming-fishing'):
            Occupation = 4
        elif (Occupation == 'Machine-op-inspct'):
            Occupation = 6
        elif (Occupation == 'Tech-support'):
            Occupation = 12
        elif (Occupation == 'Protective-serv'):
            Occupation = 10
        elif (Occupation == 'Armed-Forces'):
            Occupation = 1
        else:
            Occupation = 8




        Relationship=request.form['Relationship']
        if(Relationship=='Not-in-family'):
            Relationship=1
        elif (Relationship == 'Husband'):
            Relationship = 0
        elif (Relationship == 'Wife'):
            Relationship = 5
        elif (Relationship == 'Own-child'):
            Relationship = 3
        elif (Relationship == 'Unmarried'):
            Relationship = 4
        else:
            Relationship = 2



        Race=request.form['Race']
        if(Race=='White'):
            Race= 4
        elif(Race=='Black'):
            Race= 2
        elif (Race == 'Asian-Pac-Islander'):
            Race= 1
        elif (Race == 'Amer-Indian-Eskimo'):
            Race= 0
        else:
            Race = 3



        Sex=request.form['Sex']
        if(Sex=='Male'):
            Sex=1
        else:
            Sex=0



        Capital_Gain=int(request.form['Capital_Gain'])

        Capital_Loss=int(request.form['Capital_Loss'])

        Hours_Per_Week= int(request.form['Hours_Per_Week'])

        Country = request.form['Country']
        if(Country=='United-States'):
            Country=38
        elif(Country=='Cuba'):
            Country=4
        elif (Country == 'Jamaica'):
            Country = 22
        elif (Country == 'India'):
            Country = 18
        elif (Country == 'Mexico'):
            Country = 25
        elif (Country == 'South'):
            Country = 34
        elif (Country == 'Puerto-Rico'):
            Country = 32
        elif (Country == 'Honduras'):
            Country = 15
        elif (Country == 'England'):
            Country = 8
        elif (Country == 'Canada'):
            Country = 1
        elif (Country == 'Germany'):
            Country = 10
        elif (Country == 'Iran'):
            Country = 19
        elif (Country == 'Philippines'):
            Country = 29
        elif (Country == 'Italy'):
            Country = 21
        elif (Country == 'Poland'):
            Country = 30
        elif (Country == 'Columbia'):
            Country = 3
        elif (Country == 'Cambodia'):
            Country = 0
        elif (Country == 'Thailand'):
            Country = 36
        elif (Country == 'Ecuador'):
            Country = 6
        elif (Country == 'Laos'):
            Country = 24
        elif (Country == 'Taiwan'):
            Country = 35
        elif (Country == 'Haiti'):
            Country = 13
        elif (Country == 'Portugal'):
            Country = 31
        elif (Country == 'Dominican-Republic'):
            Country = 5
        elif (Country == 'El-Salvador'):
            Country = 7
        elif (Country == 'France'):
            Country = 9
        elif (Country == 'Guatemala'):
            Country = 12
        elif (Country == 'China'):
            Country = 2
        elif (Country == 'Japan'):
            Country = 23
        elif (Country == 'Yugoslavia'):
            Country = 40
        elif (Country == 'Outlying-US(Guam-USVI-etc)'):
            Country =27
        elif (Country == 'Scotland'):
            Country =33
        elif (Country == 'Trinadad&Tobago'):
            Country =37
        elif (Country == 'Greece'):
            Country =11
        elif (Country == 'Nicaragua'):
            Country =26
        elif (Country == 'Vietnam'):
            Country =39
        elif (Country == 'Hong'):
            Country = 16
        elif (Country == 'Ireland'):
            Country = 20
        elif (Country == 'Hungary'):
            Country = 17
        elif(Country=='Holand-Netherlands'):
            Country=14
        else:
            Country =28






        data=[Age, Work_Class, Final_Weight, Education, Marital_Status, Occupation, Relationship, Race,
              Sex, Capital_Gain, Capital_Loss, Hours_Per_Week, Country]

        feature_value=[np.array(data)]

        features_names=['Age', 'Work_Class', 'Final_Weight', 'Education', 'Marital_Status', 'Occupation',
                        'Relationship', 'Race', 'Sex', 'Capital_Gain', 'Capital_Loss', 'Hours_Per_Week',
                        'Country']

        df=pd.DataFrame(feature_value,columns=features_names)

        std_data=scaler.transform(df)

        my_predict=model.predict(std_data)

        if my_predict == 1:
            return render_template('index.html', prediction_text=f"Adult census prediction is more than 50k")
        if my_predict == 0:
            return render_template('index.html', prediction_text=f"Adult census prediction is less than or equal to 50k")


if __name__ == '__main__':
    app.run(port=8000)













