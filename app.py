from flask import Flask, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score

app = Flask(__name__)

# Load data
gold_data = pd.read_csv('Dataset/gld_price_data.csv')

# Split into X and Y
X = gold_data.drop(['Date', 'GLD'], axis=1)
Y = gold_data['GLD']

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=2)

# Train models
models = {
    'RandomForest': RandomForestRegressor(),
    'LinearRegression': LinearRegression(),
    'SVR': SVR()
}

predictions = {}

for model_name, model in models.items():
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    score = r2_score(Y_test, pred)
    predictions[model_name] = {'prediction': pred, 'score': score}

@app.route('/')
def index():
    return render_template('index.html', gold_data=gold_data, predictions=predictions)

if __name__ == '__main__':
    # Use the PORT environment variable if available, otherwise default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
