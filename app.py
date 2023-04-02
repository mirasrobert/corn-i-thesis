from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pickle

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

N_LSTM = pickle.load(open('models/N_LSTM.pkl', 'rb'))
K_LSTM = pickle.load(open('models/K_LSTM.pkl', 'rb'))
P_LSTM = pickle.load(open('models/P_LSTM.pkl', 'rb'))
PH_LSTM = pickle.load(open('models/PH_LSTM.pkl', 'rb'))


@app.route('/predict/nitrogen',methods=['POST'])
@cross_origin()
def predict_nitrogen():
    data = request.get_json(force=True)
    model_prediction = N_LSTM.predict([data])
    model_prediction = model_prediction.tolist()
    return jsonify(prediction=model_prediction[0])

@app.route('/predict/potassium',methods=['POST'])
@cross_origin()
def predict_potassium():
    data = request.get_json(force=True)
    model_prediction = K_LSTM.predict([data])
    model_prediction = model_prediction.tolist()
    return jsonify(prediction=model_prediction[0])


@app.route('/predict/phosphorus',methods=['POST'])
@cross_origin()
def predict_phosphorus():
    data = request.get_json(force=True)
    model_prediction = P_LSTM.predict([data])
    model_prediction = model_prediction.tolist()
    return jsonify(prediction=model_prediction[0])

@app.route('/predict/ph',methods=['POST'])
@cross_origin()
def predict_ph():
    data = request.get_json(force=True)
    model_prediction = PH_LSTM.predict([data])
    model_prediction = model_prediction.tolist()
    return jsonify(prediction=model_prediction[0])


if __name__ == "__main__":
    app.run(debug=True)     