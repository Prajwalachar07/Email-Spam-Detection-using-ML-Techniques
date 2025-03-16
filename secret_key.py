from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to the Spam Detection System!"

@app.route("/test")
def test():
    return "This is the test page!"

if __name__ == "__main__":
    print("Available Routes:")
    print(app.url_map)  # This prints all available routes
    app.run(debug=True)
