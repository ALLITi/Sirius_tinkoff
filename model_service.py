from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__)

# Загрузка модели и токенизатора
model = GPT2LMHeadModel.from_pretrained("C:/Users/genus/OneDrive/Desktop/Sirius_Tink/custom_ruDialoGPT/")
tokenizer = GPT2Tokenizer.from_pretrained("tinkoff-ai/ruDialoGPT-medium")

@app.route('/generate', methods=['POST'])
def generate_response():
    data = request.json
    user_input = data["input"]
    
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)