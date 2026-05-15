import json
import os
import re
from datetime import date

from dotenv import load_dotenv
from openai import OpenAI

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from intent_router import detect_calculation_intent
from guardrails import detect_policy_evasion, guardrail_response
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

EVAL_DATE = "2026-05-15"


def load_questions(path="evaluation_questions.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def retrieve(query, top_k=3):
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


def build_calculation_result(query, intent):
    if intent == "annual_leave":
        start_date = extract_date(query)

        if start_date:
            return calculate_annual_leave_days(
                start_date_str=start_date,
                today_str=EVAL_DATE,
            )

    if intent == "parental_leave":
        child_birth_date = extract_date(query)

        if child_birth_date:
            return calculate_parental_leave_end_date(
                child_birth_date_str=child_birth_date,
                today_str=EVAL_DATE,
            )

        return calculate_parental_leave_end_date(
            child_birth_date_str=None,
            today_str=EVAL_DATE,
        )

    if intent == "rent_subsidy":
        rental_days = extract_first_number(query)

        if rental_days:
            return calculate_rent_subsidy(rental_days=rental_days)

    return None


def build_calculation_text(result):
    if not result:
        return "無"

    if result["type"] == "annual_leave":
        return f"""
到職日期：{result['start_date']}
目前日期：{result['today']}
年資：{result['years']} 年 {result['months']} 個月
特休天數：{result['annual_leave_days']} 日
"""

    if result["type"] == "parental_leave":
        if result["child_birth_date"] is None:
            return f"""
目前日期：{result['today']}
從今天起最長二年：{result['max_by_two_years']}
未提供子女出生日期，無法判斷子女滿三歲限制。
"""

        return f"""
目前日期：{result['today']}
子女出生日期：{result['child_birth_date']}
從今天起最長二年：{result['max_by_two_years']}
子女滿三歲前一日：{result['max_by_child_age']}
最晚可請至：{result['end_date']}
"""

    if result["type"] == "rent_subsidy":
        return f"""
每月補助金額：{result['monthly_subsidy']} 元
租賃日數：{result['rental_days']} 日
補助金額：{result['subsidy']} 元
"""

    return "無"


def call_llm(prompt):
    if not OPENAI_API_KEY:
        return ""

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )

    return response.output_text


def answer_gpt_only(question):
    prompt = f"""
你是人資規章問答助理。請回答使用者問題。

使用者問題：
{question}
"""
    return call_llm(prompt)


def answer_rag_only(question, context):
    prompt = f"""
你是新北捷運行政與人資規章問答助理。
請只根據檢索到的規章內容回答。
如果資料不足，請回答「根據目前檢索到的規章內容，無法確認。」

使用者問題：
{question}

檢索到的規章內容：
{context}
"""
    return call_llm(prompt)


def answer_wangefficient(question, context):
    if detect_policy_evasion(question):
        return guardrail_response(), "blocked"

    intent = detect_calculation_intent(question)
    calculation_result = build_calculation_result(question, intent)
    calculation_text = build_calculation_text(calculation_result)

    prompt = f"""
你是 WangEfficient，新北捷運行政與人資規章問答助理。

你必須嚴格遵守以下規則：
1. 只能根據檢索到的規章內容回答。
2. 禁止使用外部知識。
3. 禁止自行推測。
4. 如果資料不足，請回答：「根據目前檢索到的規章內容，無法確認。」
5. 若已有確定性計算結果，禁止修改數字或日期。

使用者問題：
{question}

確定性計算結果：
{calculation_text}

檢索到的規章內容：
{context}
"""

    if OPENAI_API_KEY:
        return call_llm(prompt), intent

    return calculation_text, intent


def normalize_text(text):

    text = text.lower()

    # 去除空格與換行
    text = text.replace(" ", "")
    text = text.replace("\n", "")

    # 去除英文與中文逗號
    text = text.replace(",", "")
    text = text.replace("，", "")

    # 去除句號
    text = text.replace(".", "")
    text = text.replace("。", "")

    # 去除單位
    text = text.replace("元", "")
    text = text.replace("日", "")
    text = text.replace("天", "")

    # 中文數字簡單統一
    text = text.replace("兩", "2")
    text = text.replace("二", "2")

    return text

def contains_expected(answer, expected_contains):

    if not answer:
        return False

    normalized_answer = normalize_text(answer)

    matched = 0

    for item in expected_contains:

        normalized_item = normalize_text(item)

        if normalized_item in normalized_answer:
            matched += 1

    # 命中任一關鍵資訊即可
    return matched >= 1


def evaluate_method_answer(answer, expected_contains):
    return 1 if contains_expected(answer, expected_contains) else 0


def main():
    questions = load_questions()
    rows = []

    scores = {
        "gpt_only": 0,
        "rag_only": 0,
        "wangefficient": 0,
    }

    for q in questions:
        qid = q["id"]
        question = q["question"]
        expected_contains = q["expected_contains"]
        expected_intent = q["expected_intent"]

        results = retrieve(question, top_k=3)
        context = build_context(results)

        gpt_answer = answer_gpt_only(question)
        rag_answer = answer_rag_only(question, context)
        wang_answer, detected_intent = answer_wangefficient(question, context)

        gpt_score = evaluate_method_answer(gpt_answer, expected_contains)
        rag_score = evaluate_method_answer(rag_answer, expected_contains)
        wang_score = evaluate_method_answer(wang_answer, expected_contains)

        intent_correct = 1 if detected_intent == expected_intent else 0

        scores["gpt_only"] += gpt_score
        scores["rag_only"] += rag_score
        scores["wangefficient"] += wang_score

        rows.append({
            "id": qid,
            "type": q["type"],
            "question": question,
            "expected": " / ".join(expected_contains),
            "detected_intent": detected_intent,
            "intent_correct": intent_correct,
            "gpt_only_score": gpt_score,
            "rag_only_score": rag_score,
            "wangefficient_score": wang_score,
            "wangefficient_answer": wang_answer.replace("\n", " ")[:300],
            "gpt_only_answer": gpt_answer.replace("\n", " ")[:300],
            "rag_only_answer": rag_answer.replace("\n", " ")[:300]
        })

        print(f"[{qid}] {question}")
        print(f"  Intent: {detected_intent} | Expected: {expected_intent}")
        print(f"  GPT-only: {gpt_score} | RAG-only: {rag_score} | WangEfficient: {wang_score}")
        print("-" * 80)

    total = len(questions)

    print("\n========== Evaluation Summary ==========")
    print(f"Total Questions: {total}")
    print(f"GPT-only Accuracy: {scores['gpt_only']}/{total} = {scores['gpt_only'] / total:.2%}")
    print(f"RAG-only Accuracy: {scores['rag_only']}/{total} = {scores['rag_only'] / total:.2%}")
    print(f"WangEfficient Accuracy: {scores['wangefficient']}/{total} = {scores['wangefficient'] / total:.2%}")

    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("\nSaved results to evaluation_results.json")


if __name__ == "__main__":
    main()