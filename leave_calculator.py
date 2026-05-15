from datetime import date, datetime
from dateutil.relativedelta import relativedelta


def parse_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def calculate_years_of_service(start_date, today=None):
    if today is None:
        today = date.today()

    total_months = (
        (today.year - start_date.year) * 12
        + today.month
        - start_date.month
    )

    if today.day < start_date.day:
        total_months -= 1

    years = total_months // 12
    remaining_months = total_months % 12

    return years, remaining_months, total_months


def calculate_annual_leave_days(start_date_str, today_str=None):
    start_date = parse_date(start_date_str)

    if today_str:
        today = parse_date(today_str)
    else:
        today = date.today()

    years, months, total_months = calculate_years_of_service(start_date, today)

    if total_months < 6:
        leave_days = 0
    elif total_months < 12:
        leave_days = 3
    elif total_months < 24:
        leave_days = 7
    elif total_months < 36:
        leave_days = 10
    elif total_months < 60:
        leave_days = 14
    elif total_months < 120:
        leave_days = 15
    else:
        full_years = total_months // 12
        leave_days = min(15 + (full_years - 9), 30)

    return {
        "type": "annual_leave",
        "start_date": start_date,
        "today": today,
        "years": years,
        "months": months,
        "total_months": total_months,
        "annual_leave_days": leave_days,
    }


def calculate_parental_leave_end_date(child_birth_date_str=None, today_str=None):
    if today_str:
        today = parse_date(today_str)
    else:
        today = date.today()

    max_by_two_years = today + relativedelta(years=2)

    if child_birth_date_str:
        child_birth_date = parse_date(child_birth_date_str)
        child_three_years_old = child_birth_date + relativedelta(years=3)
        max_by_child_age = child_three_years_old - relativedelta(days=1)
        end_date = min(max_by_two_years, max_by_child_age)

        return {
            "type": "parental_leave",
            "today": today,
            "child_birth_date": child_birth_date,
            "max_by_two_years": max_by_two_years,
            "max_by_child_age": max_by_child_age,
            "end_date": end_date,
        }

    return {
        "type": "parental_leave",
        "today": today,
        "child_birth_date": None,
        "max_by_two_years": max_by_two_years,
        "max_by_child_age": None,
        "end_date": None,
    }


def calculate_rent_subsidy(rental_days, monthly_subsidy=4000, actual_rent=None):
    subsidy = round(monthly_subsidy * (rental_days / 30))

    if actual_rent is not None:
        subsidy = min(subsidy, actual_rent)

    return {
        "type": "rent_subsidy",
        "monthly_subsidy": monthly_subsidy,
        "rental_days": rental_days,
        "actual_rent": actual_rent,
        "subsidy": subsidy,
    }