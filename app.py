from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Load pre-trained model and tokenizer
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Load conversation template
with open("conversation_template.txt", "r") as file:
    conversation_template = file.read()

MAX_INPUT_LENGTH = 1024  # GPT-2 model maximum input length

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = get_ai_response(user_input)
    return jsonify({'response': response})

def get_ai_response(prompt):
    # Append the user input to the template
    conversation = conversation_template + f"\nUser: {prompt}\nAI:"

    # Encode the conversation
    inputs = tokenizer.encode(conversation, return_tensors="pt")

    # Check if the input length exceeds the maximum input length
    if inputs.size(1) > MAX_INPUT_LENGTH:
        # Calculate the excess length
        excess_length = inputs.size(1) - MAX_INPUT_LENGTH
        # Trim the beginning of the conversation to fit the maximum input length
        conversation = conversation.split("\n")
        while len(tokenizer.encode("\n".join(conversation) + f"\nUser: {prompt}\nAI:", return_tensors="pt")[0]) > MAX_INPUT_LENGTH:
            conversation = conversation[1:]  # Remove the oldest part of the conversation
        conversation = "\n".join(conversation) + f"\nUser: {prompt}\nAI:"
        inputs = tokenizer.encode(conversation, return_tensors="pt")

    attention_mask = torch.ones_like(inputs)  # Create attention mask

    try:
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=150,  # Generate up to 150 new tokens
            num_return_sequences=1,
            no_repeat_ngram_size=2,  # Avoid repetition
            temperature=0.7,  # Control randomness
            top_p=0.9,  # Nucleus sampling
            top_k=50,  # Top-k sampling
            do_sample=True  # Enable sampling
        )
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Split the response to extract the AI part
        response_text = response_text.split("AI:")[-1].strip()
    except Exception as e:
        response_text = f"An error occurred: {e}"

    # Fallback mechanism if response is irrelevant
    if not is_relevant_response(response_text):
        response_text = "I'm sorry, I didn't quite understand that. Can you please rephrase your question or ask about something specific related to the museum?"

    return response_text

def is_relevant_response(response):
    # Simple relevance check (could be enhanced with more complex logic)
    relevant_keywords = [
        "mona lisa", "leonardo da vinci", "painting", "artist", "museum", "art", 
        "history", "renaissance", "louvre", "sfumato", "famous", "masterpiece",
        "vitruvian man", "last supper", "lady with an ermine"
    ]
    return any(keyword in response.lower() for keyword in relevant_keywords)

if __name__ == '__main__':
    app.run(debug=True)
