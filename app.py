import gradio as gr
import os
import traceback
from huggingface_hub import InferenceClient

# Nexa AI Branding & Rules
SYSTEM_PROMPT = (
    "You are 'nexa. AI', an extremely intelligent, helpful, and advanced AI assistant developed by BharatCloudTechnologies, "
    "founded by CEO and co-founder Mr. Anik Kesarwani. "
    "Rule 1: Always identify yourself clearly as 'nexa. AI' in your responses when relevant, maintaining a highly professional and premium assistant tone. "
    "Rule 2: You must always reply in the exact same language as the user's question. If the user asks in Hindi, reply in clear Hindi. If they ask in Hinglish, reply in Hinglish. If they ask in English, reply in English. "
    "Rule 3: Keep your answers accurate, well-structured, and highly readable."
)

# Render ke Environment Variables se token uthana
HF_TOKEN = os.getenv("HF_TOKEN", "")

# FIX: Ab provider hardcode nahi hai (auto routing use hogi), aur
# agar pehla model kisi provider par available nahi hai to code
# automatically agle model par fallback kar lega.
MODEL_CANDIDATES = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "HuggingFaceH4/zephyr-7b-beta",
]


def _build_messages(system_prompt, message, history_list):
    messages = [{"role": "system", "content": system_prompt}]
    for chat in history_list:
        if isinstance(chat, dict):
            messages.append(chat)
        elif isinstance(chat, (list, tuple)) and len(chat) == 2:
            messages.append({"role": "user", "content": chat[0]})
            messages.append({"role": "assistant", "content": chat[1]})
    messages.append({"role": "user", "content": message})
    return messages


# 1. Core logic jo LLM se reply lati hai (Flutter ke liye hidden API endpoint)
def get_ai_response(message, history_list):
    messages = _build_messages(SYSTEM_PROMPT, message, history_list)
    last_error = None

    # FIX: client creation ab try ke andar hai, provider specify nahi kiya
    # (default "auto" route karega jo bhi provider us model ko serve karta hai)
    client = InferenceClient(token=HF_TOKEN if HF_TOKEN else None)

    for model_id in MODEL_CANDIDATES:
        try:
            completion = client.chat_completion(
                messages,
                model=model_id,
                max_tokens=512,
                stream=False,
                temperature=0.7,
                top_p=0.95,
            )
            return completion.choices[0].message.content
        except Exception as e:
            last_error = e
            print(f"=== NEXA AI: model '{model_id}' failed, trying next ===")
            traceback.print_exc()
            continue

    # Agar sab models fail ho gaye, to poora error Render logs me already
    # print ho chuka hai (upar wale loop me), yahan sirf user ko batao
    return f"Nexa AI Error: {str(last_error)}"


# 2. Gradio Web UI ke liye streaming generator
def respond(message, history, system_message, max_tokens, temperature, top_p):
    messages = [{"role": "system", "content": system_message}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    client = InferenceClient(token=HF_TOKEN if HF_TOKEN else None)
    last_error = None

    for model_id in MODEL_CANDIDATES:
        try:
            response = ""
            got_any_chunk = False
            for chunk in client.chat_completion(
                messages,
                model=model_id,
                max_tokens=max_tokens,
                stream=True,
                temperature=temperature,
                top_p=top_p,
            ):
                choices = chunk.choices
                if len(choices) and choices[0].delta.content:
                    got_any_chunk = True
                    response += choices[0].delta.content
                    yield response
            if got_any_chunk:
                return
        except Exception as e:
            last_error = e
            print(f"=== NEXA AI (respond): model '{model_id}' failed, trying next ===")
            traceback.print_exc()
            continue

    yield f"Nexa AI Error: {str(last_error)}"


# Custom Dark CSS
custom_css = """
body, .gradio-container { background-color: #0d0e12 !important; font-family: 'Inter', sans-serif; color: white !important; }
#header-container { text-align: center; padding: 25px 0; }
#logo-text { font-size: 50px; font-weight: 800; color: #ffffff; letter-spacing: -1.5px; margin: 0; display: inline-block; }
#logo-dot { color: #00dfb2; display: inline-block; animation: pulse 1.5s infinite; }
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
#footer-text { text-align: center; color: #52525b; font-size: 13px; padding: 20px 0; margin-top: 20px; border-top: 1px solid #1f2029; }
#footer-text a { color: #00dfb2; text-decoration: none; font-weight: 600; }
"""

with gr.Blocks(css=custom_css) as demo:
    with gr.Row(elem_id="header-container"):
        gr.HTML("<div><h1 id='logo-text'>nexa<span id='logo-dot'>.</span> AI</h1><p style='color: #71717a; margin-top: 5px;'>Next-Gen Intelligent Assistant</p></div>")

    # Web Chat UI
    chatbot = gr.ChatInterface(
        respond,
        type="messages",
        textbox=gr.Textbox(placeholder="Nexa AI se kuch bhi puchiye...", container=False, scale=7),
        additional_inputs=[
            gr.Textbox(value=SYSTEM_PROMPT, label="System message", visible=False),
            gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
            gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
            gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p"),
        ],
    )

    # --- FLUTTER DEDICATED API ENDPOINT ---
    user_msg_input = gr.Textbox(visible=False)
    history_input = gr.State(value=[])
    api_output = gr.Textbox(visible=False)

    api_btn = gr.Button("API", visible=False)
    api_btn.click(
        fn=get_ai_response,
        inputs=[user_msg_input, history_input],
        outputs=api_output,
        api_name="chat"
    )

    gr.HTML("<div id='footer-text'>Created by <a href='#' target='_blank'>BharatCloudTechnologies</a> &amp; CEO and co-founder <b>Mr. Anik Kesarwani</b></div>")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
