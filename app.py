from flask import Flask, request, jsonify, render_template
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained tokenizer and model
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

# Bot name
BOT_NAME = "Chalice"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_response():
    utterance = request.form.get('input_text', '')

    # Add the bot's name to the beginning of the input
    input_text_with_name = f"Your name is {BOT_NAME}: {utterance}"

    # Tokenize the modified input
    inputs = tokenizer(input_text_with_name, return_tensors="pt")

    # Passing through the utterances to the Blenderbot model
    res = model.generate(**inputs)

    # Decoding the model output
    response_text = tokenizer.decode(res[0])

    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
