from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request
from sklearn.pipeline import Pipeline
import pickle
from datetime import datetime
import pandas 
import base64
import json

from utils import get_data, get_most_recent_model


app = Flask(__name__)

model_file = get_most_recent_model()
if model_file:
    with open(model_file, "rb") as file:
        model = pickle.load(file)


@app.route("/train", methods = ["GET"])
def train():
    msg_train, msg_test, Y_train, Y_test = get_data("sms_spam.csv")
    
    pipe = Pipeline([
            ('count_vectorizer', CountVectorizer(lowercase=True, analyzer='char', stop_words='english', ngram_range = (4,4))), 
            ("classifier",MultinomialNB(alpha=0.01))
            ])
    pipe.fit(msg_train, Y_train)
    
    current_time = datetime.now().strftime("%Y-%m-%d")
    with open("models/model_mnb_{}.pickle".format(current_time),"wb") as file:
        pickle.dump(pipe, file)
        
    print(pipe.predict(msg_test))
    global model
    model = pipe
    return "training has been complete"



@app.route("/predict", methods = ["GET"])
def predict():
    msg = request.args.get("data")
    prediction = model.predict([msg])
    if prediction == 1:
        result = "not spam"
    else:
        result = "spam"
    return "the result for the predicition is: {}".format(result)

@app.route("/show_results", methods = ["GET"])
def show_results():
    model_name = request.args.get("model")
    if model_name == None:
        _, msg_test, _, Y_test = get_data("sms_spam.csv")
        predicitons = model.predict(msg_test)
        return classification_report(predicitons, Y_test)  
    
    with open(model_name, "rb") as file:
        model_testing = pickle.load(file)
    predicitons = model_testing.predict(msg_test)
    return classification_report(predicitons, Y_test)    


@app.route("/add_train_data", methods = ["GET"])
def add_train_data():
    data = request.args.get("data")
    data = base64.b64decode(data)
    print(data)
    data = data.decode("utf-8")
    data = json.loads(data)
    print(data)
    row = pandas.DataFrame(data)
    print("Feature webapi branch")
    row = row[["type", "text"]]
    row.to_csv('sms_spam.csv', mode='a', header=False, index = False)
    
    return "The data was added successfully"

app.run(host = "localhost", port = 9234)
