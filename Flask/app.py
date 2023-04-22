from flask import Flask, render_template, request
from keras.models import load_model
import pickle
import tensorflow as tf

app = Flask(__name__)

with open('count_vec.pkl', 'rb') as file:
    cv = pickle.load(file)
cla = load_model('mymodel.h5')
cla.compile(optimizer='adam', loss='binary_crossentropy')

@app.route('/')
def home():
    return render_template('index.html', ypred=None)

@app.route('/tpredict', methods=['POST'])
def predict():
    topic = request.form['tweet']
    print("Hey " + topic)
    topic = cv.transform([topic])
    print("\n" + str(topic.shape) + "\n")
    y_pred = cla.predict(topic)
    print("pred is " + str(y_pred))
    if y_pred > 0.5:
        ypred = "Positive Review"
    else:
        ypred = "Negative Review"
    return render_template('index.html', ypred=ypred)

if __name__ == "__main__":
    app.run(debug=True)