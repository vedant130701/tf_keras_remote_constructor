import flask
from script_to_model import main_script_to_model

app = flask.Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def handle_request():
    return "Flask Server & Android are Working Successfully"

@app.route('/processjson', methods=['POST'])
def processjson():
	return jsonify({'result': 'Successfully'})
#### basically when u get the request execute the following code:
'''
file = open(model_name+'.txt', 'w')
file.writelines(model_data_received_from_the_user)
file.close()
output = main_script_to_model(path, model_name)
'''

app.run(host="0.0.0.0", port=5000, debug=True)