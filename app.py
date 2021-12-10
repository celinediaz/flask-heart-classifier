
from flask import Flask, render_template, request
import pandas
import model

app = Flask(__name__)

loaded_model, scaler = model.loadFromJSON("model")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = request.form['age']
    sex = request.form['sex']
    cpt = request.form['cpt']
    bp = request.form['bp']
    ch = request.form['ch']
    fbs = request.form['fbs']
    ekg = request.form['ekg']
    hr = request.form['hr']
    ea = request.form['ea']
    std = request.form['std']
    sos = request.form['sos']
    nvf = request.form['nvf']
    th = request.form['th']
    prediction = model.predictByFeatures(pandas.DataFrame([[age, sex, cpt, bp, ch, fbs, ekg, hr, ea, std, sos, nvf, th]]), loaded_model, scaler)
    if(prediction==1): print("Presence")
    else: print("Absence")
    return render_template('result.html', data=prediction)

if __name__ == "__main__":
    app.run(debug=True)
