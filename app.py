from flask import Flask ,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def initial():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features).reshape(1,10)]
    prediction=model.predict_proba(final)
    if prediction==1:
        return render_template('index.html',pred='You have a high chance of contracting cardiovascular disease. You should start exercising')
    else:
        return render_template('index.html',pred='You have a low chance of contracting cardiovascular disease. But dont forget to exercise')
if __name__=='main':
    app.run()
