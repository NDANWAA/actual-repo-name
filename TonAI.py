import openai
from flask import Flask, request, jsonify

app = Flask(__name__)

# Set your OpenAI API key here
openai.api_key = os.getenv("API_TOKEN")

# Define the chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json  # Get the JSON data from the request
    user_message = data["message"]  # Extract the user's message

    # Use the OpenAI ChatCompletion API to get a response from GPT-3.5
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the GPT-3.5 turbo model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},  # System message to set the context
            {"role": "user", "content": user_message}  # The user's message
        ]
    )

    # Extract the response message from the API response
    reply = response["choices"][0]["message"]["content"]

    # Return the reply as a JSON response
    return jsonify({"reply": reply})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
