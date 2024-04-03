import random
import gradio as gr
import openai
import time

client = openai.Client(api_key="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")


def generate_assisted_response(messages: list) -> str:
    response = client.chat.completions.create(
    model='ft:gpt-3.5-turbo-1106:personal:chefassistant:99dTMrEA',
    messages=messages,    
    temperature=0.5,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response.choices[0].message.content

def generate_messages(messages: list, query: str) -> list:
    formated_messages = [
        {
            'role': 'system',
            'content': 'You are a recipe suggesting chatbot and your role is to help chef through the recipes and also to suggest new recipes based on the ingredients. Do not reply any other questions if they are not related to cooking or food.'
        }
    ]
    for m in messages:
        formated_messages.append({
            'role': 'user',
            'content': m[0]
        })
        formated_messages.append({
            'role': 'assistant',
            'content': m[1]
        })
    formated_messages.append(
        {
            'role': 'user',
            'content': query
        }
    )
    return formated_messages

def generate_response(query: str, chat_history: list) -> tuple:
        messages = generate_messages(chat_history, query)
        bot_message = generate_assisted_response(messages)
        chat_history.append((query, bot_message))
        time.sleep(random.randint(0, 5))
        return '', chat_history

with gr.Blocks() as demo:

    chatbot = gr.Chatbot(label='Openai Chatbot', height=550)
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    
    msg.submit(generate_response, [msg, chatbot], [msg, chatbot])

demo.launch(share=True)
