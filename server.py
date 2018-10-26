import os
from flask import Flask, request, jsonify
from nlunetwork import runtime

app = Flask(__name__)

default_model_path = 'nlunetwork/results/train_all/conf_4/huric_eb/with_framenet/'
model_path = os.environ.get('MODEL_PATH', default_model_path)

if not os.path.isfile(model_path + 'model_fold_0.ckpt.meta'):
    raise FileNotFoundError('Check that model files exist in the folder ' + model_path)

nlu = runtime.NLUWrapper(model_path)

@app.route('/')
def hello():
    return 'hello world'

@app.route('/nlu', methods=['GET', 'OPTIONS'])
def nlu_endpoint():
    text = request.args.get('text')
    if not text:
        return 'use "text" url param'
    nlu_output = nlu.parse(text)
    return jsonify({
        'text': text,
        'nlu': nlu_output
    })