import pickle
from flask import Flask, request
from flask_cors import CORS, cross_origin
import pandas as pd

# Declare the flask app
app = Flask(__name__)

# Enable cross origin request for our application
cors = CORS(app)

model = pickle.load(open("./Output/model.pkl", "rb"))

# Enable an api route for status check
@app.route('/check', methods= ['GET'])
@cross_origin()
def return_status():
    return "Yay! Flask App is running"

# Enable api route to get time series predictions
@app.route('/', methods = ['POST'])
@cross_origin()
def return_model_prediction():
        # Get prediction results and respond
        # POST Request: Where we need csv file
    try:
        data = pd.read_csv(request.files.get("data"))
        data["timestamp"] = data["month"].apply(lambda x : x.timestamp())
        predictions = model.predict(data["timestamp"].values.reshape(-1, 1))
        final_predictions = list(predictions)
        return {"status_code":200,"message":"Sucess", "body": {"preds": final_predictions}}

    except Exception as e:
        print(f"Error occured :     {e}")
        return {"status_code":404, "message": f"Error :-    {e}"}

if __name__ == '__main__':
    app.run("0.0.0.0", port= 5000)
