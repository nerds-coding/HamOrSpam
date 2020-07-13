from flask import Flask, jsonify, request, make_response, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


app = Flask(__name__)

spam = pd.read_csv('HamSpamFinalData.csv', encoding='ISO-8859-1')
spam.dropna(inplace=True)
global vector
global model
vector = TfidfVectorizer(max_features=500, ngram_range=(1, 4), lowercase=True)
vals = vector.fit_transform(spam['ppData'])
model = pickle.load(open('Adaboostlog.sav', 'rb'))



def ModelPrediction(value):
    msg = vector.transform([str(value)])
    msg = model.predict(msg)[0]
    if(msg == 0):
        return "NOT SPAM"
    else:
        return "SPAM"


@app.route("/")
def hello():
    return render_template('index.html')

# --------------------------------- JSON PREDICTION -------------------------
@app.route("/json", methods=["POST"])
def jsonFile():
    if(request.is_json):
        req = request.get_json()
        msg = req.get("message")

        if(req!=None):
            response = {
                "prediction":" "+ModelPrediction(str(msg))
            }
            return make_response(jsonify(response), 200)
        else:
            response={
            "prediction":"Please Write something"
            }
            return make_response(jsonify(response),200)
    else:
        return make_response(jsonify({"prediction": "No Json Received"}), 400)


# --------------------------------- WEB PAGE PREDICTION -------------------------
@app.route("/pred",methods=["POST"])
def prediction():
    try:
        if request.method == 'POST':
            msg =request.form['message']

    except(Exception)as e:
        return ""+str(e)
    else:
        if(len(msg.strip())>0):
            return render_template('index.html',Hello=" "+ModelPrediction(msg))
        else:
            return render_template('index.html',Hello='Please Write something')





if __name__ == "__main__":
    app.run(debug=True)
