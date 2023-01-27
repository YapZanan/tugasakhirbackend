import time
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, pipeline
from flask import Flask, request, jsonify, render_template, make_response
from flask_cors import CORS, cross_origin
app = Flask(__name__)
CORS(app, support_credentials=True)


def load_model_generasi():

    model_name_dict = {
    'gpt-neo-2.7B': 'model_generasi',
    }

    model_dict = {}

    for call_name, real_name in model_name_dict.items():
        print('\tLoading model: %s' % call_name)
        model = GPTNeoForCausalLM.from_pretrained(real_name)
        tokenizer = GPT2Tokenizer.from_pretrained(real_name)
        model_dict[call_name + '_model'] = model
        model_dict[call_name + '_tokenizer'] = tokenizer

    return model_dict

def generasi(text, min_length):
    if len(model_dict) == 2:
        model_name = 'gpt-neo-2.7B'

    start_time = time.time()
    # source = flores_codes[source]
    # target = flores_codes[target]

    model = model_dict[model_name + '_model']
    tokenizer = model_dict[model_name + '_tokenizer']

    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    output = generator(text, min_length)

    end_time = time.time()

    full_output = output
    output = output[0]['generation_text']
    result = {'inference_time': end_time - start_time,
              'result': output,
              'full_output': full_output}
    return result



@app.route('/generasi', methods=['POST'])
@cross_origin()
def translate():
    data = request.get_json()
    print(request.data)  # print the request data to the console
    text = data['text']
    min_length = data['min_length']

    result = generasi(text, min_length)
    response = make_response(jsonify(result))
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    return response


if __name__ == '__main__':
    global model_dict
    model_dict = load_model_generasi()
    app.run()
