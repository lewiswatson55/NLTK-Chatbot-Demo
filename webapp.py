from chatbot import *
from flask import Flask

# create flask app

app = Flask(__name__)

@app.route('/chatbot/<query>')
def chatbot(query):
    resp = chatbot_response(query)
    return resp, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)