from flask import Flask, request, jsonify
from joint import runtime

app = Flask(__name__)
nlu = runtime.NLUWrapper()

@app.route('/')
def hello():
    return 'hello world'

@app.route('/nlu', methods=['GET', 'OPTIONS'])
def nlu_endpoint():
    text = request.args.get('text')
    nlu_output = nlu.parse(text)
    return jsonify({
        'text': text,
        'nlu': nlu_output
    })