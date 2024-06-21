import sys
import os
import numpy as np
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
import numpy as np
from PIL import Image
import pickle

rnn_dir = 'theROCK/models/rnn/'
sys.path.append(rnn_dir)
sys.path.append(os.getcwd()+'/'+rnn_dir)
sys.path.append(os.getcwd())
from DeepRouteSetHelper import sanityCheckAndOutput, plotAProblem
from model import n_values, n_a, inference_model, holdIx_to_holdStr, handStringList, predict_and_sample

with open("./theROCK/raw_data/holdStr_to_holdIx", 'rb') as f:
    holdStr_to_holdIx = pickle.load(f)
    
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', board='/static/images/moonboard.jpg')

@app.route('/predict',methods=['POST'])
def predict():
    """
    Render image of generated route
    """
    start_hold = next(request.form.values())
    start_hold = start_hold.upper()
    if np.random.choice([0, 1]):
        start_hold += '-RH'
    else:
        start_hold += '-LH'
    start_hold = holdStr_to_holdIx.get(start_hold)
    if not start_hold:
        flash('Invalid value. Please enter a valid value.')
        return redirect(url_for('home'))
    
    while True:
        x_initializer = np.zeros((1, 1, n_values))
        x_initializer = np.random.rand(1, 1, n_values) / 100
        a_initializer = np.random.rand(1, n_a) * 150
        c_initializer = np.random.rand(1, n_a) / 2
    
        results, indices = predict_and_sample(inference_model, start_hold, x_initializer, a_initializer, c_initializer)
        passCheck, outputListInString, outputListInIx = sanityCheckAndOutput(indices, 
                                                                             holdIx_to_holdStr, 
                                                                             handStringList, 
                                                                             printError=True)
        if passCheck: # and start_hold in outputListInString[0]:
            print('pass')
            print(results)
            print(outputListInString)
            # outputListInString.insert(0, start_hold)
            plotAProblem(outputListInString, key="problem", save=True)
            break
    
    image = Image.open('static/images/problem.png')
    image_box = image.getbbox()
    cropped = image.crop(image_box)
    cropped.save('static/images/problem.png')

    return render_template('index.html', board='/static/images/problem.png')

def is_valid(start_hold):
    # Check that start_hold is not too high/low and it exists

    return value and value.isdigit()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
