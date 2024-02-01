# app.py
from flask import Flask, render_template
from app.routes import app as routes_app

app = Flask(__name__)
app.register_blueprint(routes_app)

if __name__ == '__main__':
    app.run(debug=True)
