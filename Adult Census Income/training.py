from flask import Flask, request
from flask import Response
from feature_engineering.scaling_tranform import Data_scaling
from Data_Modelling.model_tuning import Model_Finder


app = Flask(__name__)

@app.route('/train',methods=['POST'])

def training():

    try:

        if request.json is not None:
            operation=request.json['operation']
            if(operation.lower()== 'training'):

                '''Creating scaling pickle file by calling Data_scaling function'''
                scaling_obj=Data_scaling()
                scaling_obj.scaling()

                '''Creating best model by calling get_best_model function by performing model training'''
                training_obj=Model_Finder()
                model_name=training_obj.get_best_model()

                return Response(f"Training successfull!!,best model is {model_name}")


    except Exception as e:

        return Response("Error Occurred! %s" % e)


if __name__ =='__main__':
    app.run(debug=True,port=5000)














