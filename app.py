import os
import re

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from guardrails import detect_policy_evasion, guardrail_response
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from intent_router import detect_calculation_intent
from leave_calculator import (
    calculate_annual_leave_days,
    calculate_parental_leave_end_date,
    calculate_rent_subsidy,
)


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "wangefficient_rules"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_vectorstore():
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=CHROMA_DIR,
    )


def retrieve(query, top_k=4):
    vectorstore = load_vectorstore()
    return vectorstore.similarity_search_with_score(query=query, k=top_k)


def build_context(results):
    context = ""

    for i, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source_file", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "unknown")

        context += f"\n[資料 {i}]\n"
        context += f"來源：{source}\n"
        context += f"Chunk ID：{chunk_id}\n"
        context += f"內容：{doc.page_content}\n"

    return context


def build_calculation_text(calculation_result):
    if not calculation_result:
        return "無"

    if calculation_result["type"] == "annual_leave":
        return f"""
系統已完成確定性計算，以下結果不可修改：

計算類型：特別休假天數
到職日期：{calculation_result['start_date']}
目前日期：{calculation_result['today']}
年資：{calculation_result['years']} 年 {calculation_result['months']} 個月
特休天數：{calculation_result['annual_leave_days']} 日

請根據上述數值回答。
禁止重新計算。
禁止修改數字。
"""

    if calculation_result["type"] == "parental_leave":
        if calculation_result["child_birth_date"] is None:
            return f"""
系統判斷此問題需要育嬰留職停薪截止日計算。

目前日期：{calculation_result['today']}
規則限制一：從今天起最長不得超過二年，日期為 {calculation_result['max_by_two_years']}
但使用者未提供子女出生日期，因此無法計算「子女滿三歲前」的限制。

請回答：需要提供子女出生日期才能精確計算最晚截止日。
"""
        return f"""
系統已完成確定性計算，以下結果不可修改：

計算類型：育嬰留職停薪截止日
目前日期：{calculation_result['today']}
子女出生日期：{calculation_result['child_birth_date']}
從今天起最長二年：{calculation_result['max_by_two_years']}
子女滿三歲前一日：{calculation_result['max_by_child_age']}
最晚可請至：{calculation_result['end_date']}

請根據上述數值回答。
禁止重新計算。
禁止修改日期。
"""

    if calculation_result["type"] == "rent_subsidy":
        return f"""
系統已完成確定性計算，以下結果不可修改：

計算類型：租屋補助不足月金額
每月補助金額：{calculation_result['monthly_subsidy']} 元
租賃日數：{calculation_result['rental_days']} 日
實際租金：{calculation_result['actual_rent']}
補助金額：{calculation_result['subsidy']} 元

請根據上述數值回答。
禁止重新計算。
禁止修改金額。
"""

    return "無"


def generate_answer(query, context, calculation_result=None):
    if not OPENAI_API_KEY:
        return "未設定 OPENAI_API_KEY"

    client = OpenAI(api_key=OPENAI_API_KEY)

    calculation_text = build_calculation_text(calculation_result)

    prompt = f"""
你是 WangEfficient，新北捷運行政與人資規章問答助理。

你必須嚴格遵守以下規則：

1. 只能根據檢索到的規章內容回答。
2. 禁止使用外部知識。
3. 禁止自行推測。
4. 如果資料不足，請回答：「根據目前檢索到的規章內容，無法確認。」
5. 回答時必須列出依據條文。
6. 若已有確定性計算結果，禁止修改數字或日期。
7. 不要說「歡迎隨時再問」這類客服結尾。

使用者問題：
{query}

確定性計算結果：
{calculation_text}

檢索到的規章內容：
{context}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )

    return response.output_text


def extract_date(query):
    match = re.search(r"\d{4}-\d{2}-\d{2}", query)
    if match:
        return match.group()
    return None


def extract_first_number(query):
    numbers = re.findall(r"\d+", query)
    if numbers:
        return int(numbers[0])
    return None


st.set_page_config(
    page_title="WangEfficient",
    page_icon="🚇",
    layout="wide",
)

st.title("🚇 WangEfficient")
st.caption("新北捷運行政與人資規章 AI 問答系統 MVP")

query = st.text_input(
    "請輸入問題",
    placeholder="例如：我 2023-08-01 到職，現在有幾天特休？",
)

top_k = st.slider(
    "檢索 Top-K",
    min_value=1,
    max_value=8,
    value=4,
)


if st.button("查詢") and query:
    with st.spinner("檢索規章中..."):
        results = retrieve(query, top_k=top_k)
        context = build_context(results)

    intent = detect_calculation_intent(query)
    st.caption(f"Intent Router：{intent}")

    calculation_result = None
    if detect_policy_evasion(query):
        st.error(guardrail_response())
        st.stop()
    if intent == "annual_leave":
        start_date = extract_date(query)

        if start_date:
            calculation_result = calculate_annual_leave_days(start_date)
        else:
            st.warning("請提供到職日期，例如：2023-08-01")

    elif intent == "parental_leave":
        child_birth_date = extract_date(query)

        if child_birth_date:
            calculation_result = calculate_parental_leave_end_date(child_birth_date)
        else:
            calculation_result = calculate_parental_leave_end_date()
            st.warning("若要精確計算育嬰假截止日，請提供子女出生日期，例如：2024-03-01")

    elif intent == "rent_subsidy":
        rental_days = extract_first_number(query)

        if rental_days:
            calculation_result = calculate_rent_subsidy(rental_days=rental_days)
        else:
            st.warning("請提供租賃天數，例如：租屋補助不足月 10 天可以領多少？")

    st.subheader("AI 回答")

    with st.spinner("產生回答中..."):
        answer = generate_answer(
            query=query,
            context=context,
            calculation_result=calculation_result,
        )

    st.write(answer)

    if calculation_result:
        st.subheader("Deterministic Calculation")

        if calculation_result["type"] == "annual_leave":
            st.success(
                f"""
到職日期：{calculation_result['start_date']}

目前日期：{calculation_result['today']}

年資：{calculation_result['years']} 年 {calculation_result['months']} 個月

目前特休天數：{calculation_result['annual_leave_days']} 日
"""
            )

        elif calculation_result["type"] == "parental_leave":
            if calculation_result["child_birth_date"] is None:
                st.warning(
                    f"""
目前日期：{calculation_result['today']}

從今天起最長二年：{calculation_result['max_by_two_years']}

但尚未提供子女出生日期，因此無法判斷子女滿三歲前的限制。
"""
                )
            else:
                st.success(
                    f"""
目前日期：{calculation_result['today']}

子女出生日期：{calculation_result['child_birth_date']}

從今天起最長二年：{calculation_result['max_by_two_years']}

子女滿三歲前一日：{calculation_result['max_by_child_age']}

最晚可請至：{calculation_result['end_date']}
"""
                )

        elif calculation_result["type"] == "rent_subsidy":
            st.success(
                f"""
每月補助金額：{calculation_result['monthly_subsidy']} 元

租賃日數：{calculation_result['rental_days']} 日

補助金額：{calculation_result['subsidy']} 元
"""
            )

    st.subheader("檢索到的依據條文")

    for i, (doc, score) in enumerate(results, start=1):
        with st.expander(
            f"Top {i} | Score: {score:.4f} | {doc.metadata.get('source_file')}"
        ):
            st.write(doc.page_content)
            st.caption(f"Chunk ID: {doc.metadata.get('chunk_id')}")