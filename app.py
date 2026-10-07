"""
Flask webhook that receives incoming WhatsApp messages from Twilio,
runs them through the RAG engine, and replies with the generated answer.

Run locally:
    python app.py
Then expose it with ngrok and point your Twilio Sandbox webhook at
https://<your-ngrok-subdomain>.ngrok.io/webhook
"""
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

from rag_engine import answer_question
import conversation_memory
import user_profiles

app = Flask(__name__)


@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    incoming_msg = request.values.get("Body", "").strip()
    sender = request.values.get("From", "unknown")

    print(f"Message from {sender}: {incoming_msg}")

    if not incoming_msg:
        reply_text = "Sorry, I didn't receive any text. Could you try again?"

    elif user_profiles.get_profile(sender) is None:
        # Brand new number — start onboarding by asking for their name.
        user_profiles.start_onboarding(sender)
        reply_text = "Hi there! I'm your wellness assistant. What's your name?"

    elif user_profiles.is_awaiting_name(sender):
        # Their previous message was our "what's your name?" prompt, so
        # treat this message as the name itself.
        name = incoming_msg
        user_profiles.save_name(sender, name)
        reply_text = f"Lovely to meet you, {name}! What can I help you with today?"

    else:
        user_name = user_profiles.get_name(sender)
        history = conversation_memory.get_history(sender)
        reply_text = answer_question(incoming_msg, history=history, user_name=user_name)
        conversation_memory.add_turn(sender, incoming_msg, reply_text)

    print(f"Generated reply ({len(reply_text)} chars): {reply_text}")

    twiml_response = MessagingResponse()
    twiml_response.message(reply_text)
    twiml_str = str(twiml_response)
    print(f"TwiML being returned: {twiml_str}")
    return twiml_str


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(debug=True, port=5000)