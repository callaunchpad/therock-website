import sys
import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image

rnn_dir = 'theROCK/models/rnn/'
sys.path.append(rnn_dir)
from DeepRouteSetHelper import sanityCheckAndOutput, plotAProblem
from model import n_values, n_a, inference_model, holdIx_to_holdStr, handStringList, predict_and_sample

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
    
    while True:
        x_initializer = np.zeros((1, 1, n_values))
        x_initializer = np.random.rand(1, 1, n_values) / 100
        a_initializer = np.random.rand(1, n_a) * 150
        c_initializer = np.random.rand(1, n_a) / 2
    
        results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
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

if __name__ == "__main__":
    app.run(debug=True)
