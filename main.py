from flask import Flask, request, jsonify, render_template
from getAnswer import get_answer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/getAnswer', methods=['POST'])
def getAnswer():
    question = request.json['question']
    answer = get_answer(question)
    return jsonify({'answer': answer})

@app.route('/home' , methods=['GET'])
def index():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=80)