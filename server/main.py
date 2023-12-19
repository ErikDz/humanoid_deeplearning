from flask import Flask, request, render_template
import requests
from api_call import prompt_gpt
app = Flask(__name__)

# Replace with the actual API endpoint you want to use
API_URL = "https://api.example.com"  # Replace with your API URL

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        prompt = request.form["prompt"]
        response = prompt_gpt(prompt)
        print(response)
        try:
            response = response.split("```python")[1].split("```")[0]
            
        except:
            pass
        return render_template("index.html", response=response)

    return render_template("index.html", response=None)


if __name__ == "__main__":
    app.run(debug=True, host=)
