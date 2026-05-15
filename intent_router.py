def detect_calculation_intent(query):
    if ("特休" in query or "特別休假" in query) and (
        "到職" in query or "幾天" in query or "年資" in query
    ):
        return "annual_leave"

    if "育嬰" in query and (
        "請到哪天" in query
        or "最多" in query
        or "最久" in query
        or "多久" in query
        or "截止" in query
    ):
        return "parental_leave"

    if "租屋補助" in query and (
        "多少" in query
        or "金額" in query
        or "不足月" in query
        or "幾天" in query
    ):
        return "rent_subsidy"

    return "rag_only"