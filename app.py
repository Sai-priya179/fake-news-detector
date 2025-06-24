from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('model/classifier.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    input_data = vectorizer.transform([news])
    prediction = model.predict(input_data)[0]
    label = "REAL" if prediction == 1 else "FAKE"
    return render_template('index.html', prediction=label, news=news)

if __name__ == '__main__':
    app.run(debug=True)