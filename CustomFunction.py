#define date function AI does not have calendar
def calendar(question):
    now = date.today()
    if "today" in question.lower():
        now = now.strftime("%d %b %Y")
        datequery = f"today refer to {now}"
    elif "yesterday" in question.lower():
        yesterday = now - timedelta(days=1)
        now = yesterday.strftime("%d %b %Y")
        datequery = f"yesterday refer to {now}"
    elif "last week" in question.lower() or "previous week" in question.lower():
        last_monday = now - timedelta(days=now.weekday()) - timedelta(days=7)
        last_sunday = last_monday + timedelta(days=6)
        start_date = last_monday.strftime("%d %b %Y")
        end_date = last_sunday.strftime("%d %b %Y")
        datequery = f"date range is from {start_date} to {end_date}"
    elif "this week" in question.lower() or "current week" in question.lower():
        this_monday = now - timedelta(days=now.weekday())
        this_sunday = this_monday + timedelta(days=6)
        start_date = this_monday.strftime("%d %b %Y")
        end_date = this_sunday.strftime("%d %b %Y")
        datequery = f"date range is from {start_date} to {end_date}"
    elif "last month" in question.lower() or "previous month" in question.lower():
        last_month = now.month - 1 if now.month > 1 else 12
        last_year = now.year if now.month > 1 else now.year - 1
        last_month_start = date(last_year, last_month, 1)
        last_month_end = date(now.year, now.month, 1) - timedelta(days=1)
        start_date = last_month_start.strftime("%d %b %Y")
        end_date = last_month_end.strftime("%d %b %Y")
        datequery = f"date range is from {start_date} to {end_date}"
    elif "this month" in question.lower() or "current month" in question.lower():
        this_month_start = date(now.year, now.month, 1)
        next_month = now.month + 1 if now.month < 12 else 1
        next_year = now.year if now.month < 12 else now.year + 1
        this_month_end = date(next_year, next_month, 1) - timedelta(days=1)
        start_date = this_month_start.strftime("%d %b %Y")
        end_date = this_month_end.strftime("%d %b %Y")
        datequery = f"date range is from {start_date} to {end_date}"
    elif "last year" in question.lower() or "previous year" in question.lower():
        last_year = now.year - 1
        last_year_start = date(last_year, 1, 1)
        last_year_end = date(last_year, 12, 31)
        start_date = last_year_start.strftime("%d %b %Y")
        end_date = last_year_end.strftime("%d %b %Y")
        datequery = f"date range is from {start_date} to {end_date}"
    else:
        return None
    return datequery