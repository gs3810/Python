from flask import Flask, render_template, jsonify
from random import sample

app = Flask(__name__)

@app.route("/")
def chart():
    return render_template('chart.html')
    
#	labels = ["January","February","March","April","May","June","July","August"]
#	values = [10,9,8,7,6,4,7,8]
#	return render_template('chart.html', values=values, labels=labels)

@app.route("/data")
def data():
	return jsonify({'results' : sample(range(1,10),2)})

if __name__ == "__main__":
	app.run(debug=False, port=5000)    