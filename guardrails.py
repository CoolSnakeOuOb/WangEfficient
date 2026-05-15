EVASION_KEYWORDS = [
    "鑽漏洞",
    "繞過",
    "規避",
    "避開",
    "不照規定",
    "不用申請",
    "不用核准",
    "不要留下紀錄",
    "怎麼說比較不會被發現",
    "有沒有其他說法",
    "能不能假裝",
    "幫我包裝",
    "主管不知道",
    "人資不知道",
    "偷請",
    "私下",
]


def detect_policy_evasion(query):
    for keyword in EVASION_KEYWORDS:
        if keyword in query:
            return True
    return False


def guardrail_response():
    return """
此問題可能涉及規避公司規章或不當利用制度漏洞，因此系統不提供規避、包裝或繞過流程的建議。

若您需要辦理請假、補休、租屋補助或其他人資事項，請依公司正式規章與申請流程辦理；我可以協助您查詢正確規定、申請條件與應備文件。
"""