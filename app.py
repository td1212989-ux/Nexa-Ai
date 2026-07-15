import gradio as gr
import os
import requests
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Nexa AI Branding & System Rules
SYSTEM_PROMPT = (
    "You are 'nexa. AI', an extremely intelligent, helpful, and advanced AI assistant developed by BharatCloudTechnologies, "
    "founded by CEO and co-founder Mr. Anik Kesarwani. "
    "Rule 1: Always identify yourself clearly as 'nexa. AI' in your responses when relevant, maintaining a highly professional and premium assistant tone. "
    "Rule 2: You must always reply in the exact same language as the user's question. If the user asks in Hindi, reply in clear Hindi. If they ask in Hinglish, reply in Hinglish. If they ask in English, reply in English. "
    "Rule 3: Keep your answers accurate, well-structured, and highly readable."
)

# Render Environment Variable se HuggingFace token uthana
HF_TOKEN = os.getenv("HF_TOKEN", "")

# NOTE: "meta-llama/Meta-Llama-3.1-8B-Instruct" was gated and not
# available on the current Inference Providers router. Using
# Qwen/Qwen2.5-7B-Instruct instead — lightweight, free, and reliably
# available. You can override this anytime via the MODEL_ID environment
# variable on Render without touching this file.
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")


#---------------------------------------------------------------------
# Core model call — shared by the Gradio UI, the REST API, and the
# browser-testable GET route below, so there's only one place that
# talks to Hugging Face.
#
# NOTE: this calls the new HF "Inference Providers" router directly via
# requests, instead of huggingface_hub's InferenceClient. The old
# api-inference.huggingface.co endpoint that InferenceClient used to
# default to has been decommissioned (that's the DNS/MaxRetryError you
# were seeing) — router.huggingface.co is the current one, and it uses
# a plain OpenAI-compatible /v1/chat/completions schema, so a simple
# requests.post is actually simpler and more future-proof than pulling
# in the client library for it.
#---------------------------------------------------------------------
HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"


def call_model(message: str, history: Optional[List[List[str]]] = None) -> str:
    formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        for user_msg, ai_msg in history:
            formatted_messages.append({"role": "user", "content": user_msg})
            formatted_messages.append({"role": "assistant", "content": ai_msg})

    formatted_messages.append({"role": "user", "content": message})

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_ID,
        "messages": formatted_messages,
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.95,
        "stream": False,
    }

    try:
        resp = requests.post(HF_ROUTER_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError:
        return f"Nexa AI Error: HTTP {resp.status_code} — {resp.text[:300]}"
    except Exception as e:
        return f"Nexa AI Error: {str(e)}"


#---------------------------------------------------------------------
# Gradio UI — kept only for manual testing in the browser at /ui.
# The Flutter app should call the /chat REST endpoint instead.
#---------------------------------------------------------------------
def respond(message):
    return call_model(message, None)


custom_css = """
body, .gradio-container {
    background-color: #0d0e12 !important;
    font-family: 'Inter', sans-serif;
    color: white !important;
}
"""

gradio_app = gr.Interface(
    fn=respond,
    inputs=gr.Textbox(placeholder="Nexa AI se kuch bhi puchiye...", label="Message"),
    outputs=gr.Textbox(label="Nexa. AI Response"),
    title="nexa. AI",
    description="Next-Gen Intelligent Assistant by BharatCloudTechnologies",
    css=custom_css,
)


#---------------------------------------------------------------------
# FastAPI app — this is what the Flutter app (and anything else) talks
# to. CORS is wide open so no origin/browser/app ever gets blocked.
#---------------------------------------------------------------------
api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[List[str]]] = None


class ChatResponse(BaseModel):
    reply: str


#---------------------------------------------------------------------
# Real endpoint for the app to call:
#   POST https://<your-app>.onrender.com/chat
#   Body: {"message": "hello", "history": []}
#   Response: {"reply": "..."}
#---------------------------------------------------------------------
@api.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    reply = call_model(req.message, req.history)
    return ChatResponse(reply=reply)


#---------------------------------------------------------------------
# Browser-testable version of the same thing — just open this URL
# directly in Chrome, no Postman/curl needed:
#   https://<your-app>.onrender.com/chat-test?message=hello
#---------------------------------------------------------------------
@api.get("/chat-test")
def chat_test(message: str = "Hello, who are you?"):
    reply = call_model(message, None)
    return {"you_asked": message, "reply": reply}


#---------------------------------------------------------------------
# Lightweight ping endpoint for UptimeRobot (or any keep-alive service)
# to hit every few minutes so Render's free tier doesn't spin the
# service down. Does NOT call the model — just confirms the server is
# alive, so it's fast and free of any Hugging Face usage.
#   https://<your-app>.onrender.com/ping
#---------------------------------------------------------------------
@api.get("/ping")
def ping():
    return {"status": "alive"}


@api.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "hf_token_set": bool(HF_TOKEN)}


# Mounts the Gradio UI at /ui, and the FastAPI routes (/chat, /chat-test,
# /ping, /health) stay at the root — both live on the same Render
# service/port.
app = gr.mount_gradio_app(api, gradio_app, path="/ui")

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
