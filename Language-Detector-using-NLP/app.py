from flask import Flask , render_template , request , redirect
import joblib as jb
import language_detection


app = Flask(__name__)


@app.route('/')
def display():
	return render_template('index.html')

# send form data/request on same route
@app.route('/' , methods=['POST'])
def marks():

	if request.method == 'POST':
		
		inp_text = request.form['text']
		lang = language_detection.predict(inp_text)

	return render_template('index.html' , pred_lang = lang)



if __name__ == '__main__':
	app.run(debug =True)