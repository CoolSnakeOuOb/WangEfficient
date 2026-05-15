import os

from flask import Flask, request, abort
from dotenv import load_dotenv

from linebot.v3 import WebhookHandler
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent

from app import (
    retrieve,
    build_context,
    generate_answer,
)

from intent_router import detect_calculation_intent
from leave_calculator import (
    calculate_annual_leave_days,
    calculate_parental_leave_end_date,
    calculate_rent_subsidy,
)
from guardrails import (
    detect_policy_evasion,
    guardrail_response,
)


# =========================
# Load Environment
# =========================

load_dotenv()

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")


# =========================
# Flask App
# =========================

app = Flask(__name__)


# =========================
# LINE SDK
# =========================

configuration = Configuration(
    access_token=LINE_CHANNEL_ACCESS_TOKEN
)

handler = WebhookHandler(LINE_CHANNEL_SECRET)


# =========================
# Deterministic Calculation
# =========================

def build_calculation_result(query, intent):

    import re

    if intent == "annual_leave":

        date_match = re.search(r"\d{4}-\d{2}-\d{2}", query)

        if date_match:

            start_date = date_match.group()

            return calculate_annual_leave_days(start_date)

    elif intent == "parental_leave":

        date_match = re.search(r"\d{4}-\d{2}-\d{2}", query)

        if date_match:

            child_birth_date = date_match.group()

            return calculate_parental_leave_end_date(
                child_birth_date
            )

    elif intent == "rent_subsidy":

        number_match = re.search(r"\d+", query)

        if number_match:

            rental_days = int(number_match.group())

            return calculate_rent_subsidy(rental_days)

    return None


# =========================
# Webhook
# =========================

@app.route("/callback", methods=["POST"])
def callback():

    signature = request.headers["X-Line-Signature"]

    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)

    except Exception as e:
        print("Webhook Error:", e)
        abort(400)

    return "OK"


# =========================
# Message Event
# =========================

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):

    query = event.message.text

    # ===== Guardrail =====

    if detect_policy_evasion(query):

        reply_text = guardrail_response()

    else:

        # ===== Retrieval =====

        results = retrieve(query, top_k=4)

        context = build_context(results)

        # ===== Intent Router =====

        intent = detect_calculation_intent(query)

        # ===== Deterministic =====

        calculation_result = build_calculation_result(
            query,
            intent
        )

        # ===== LLM =====

        reply_text = generate_answer(
            query=query,
            context=context,
            calculation_result=calculation_result,
        )

    # ===== Reply =====

    with ApiClient(configuration) as api_client:

        line_bot_api = MessagingApi(api_client)

        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[
                    TextMessage(text=reply_text[:4500])
                ],
            )
        )


# =========================
# Main
# =========================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )