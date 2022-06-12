import main as chatbot

# create flask app

@app.route('/chatbot/<query>')
def chatbot(query):
    resp = chatbot.chatbot_response(query)
    return resp, 200