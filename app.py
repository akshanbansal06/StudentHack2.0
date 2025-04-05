from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('nasaLanding.html')

@app.route('/valuationForm')
def valutationForm():
    return render_template('valuationForm.html')

if __name__ == '__main__':
    app.run(debug=True)
