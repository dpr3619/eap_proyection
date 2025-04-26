import pandas as pd
from workalendar.america import Colombia

def generate_monthly_calendar(start_year=2025, end_year=2040):
    """
    Genera un DataFrame con características mensuales de calendario para Colombia.

    Returns:
        pd.DataFrame con columnas: year, month, workdays, weekends, holidays
    """
    cal = Colombia()
    dates = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq="D")

    df = pd.DataFrame({"date": dates})
    df["Year"] = df["date"].dt.year
    df["Month"] = df["date"].dt.month
    df["YearMonth"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str), format='%Y-%m')
    df["weekday"] = df["date"].dt.weekday  # 0=Lunes, 6=Domingo
    df["is_weekend"] = df["weekday"] >= 5
    df["is_holiday"] = df["date"].apply(cal.is_holiday)
    df["is_workday"] = ~df["is_weekend"] & ~df["is_holiday"]

    monthly = df.groupby([df["date"].dt.to_period("M")]).agg({
        "is_workday": "sum",
        "is_weekend": "sum",
        "is_holiday": "sum"
    }).reset_index()

    monthly.rename(columns={
        "date": "period",
        "is_workday": "workdays",
        "is_weekend": "weekends",
        "is_holiday": "holidays"
    }, inplace=True)

    monthly["Year"] = monthly["period"].dt.year
    monthly["Month"] = monthly["period"].dt.month
    monthly["YearMonth"] = monthly["period"].dt.to_timestamp()

    return monthly[["YearMonth", "Year", "Month", "workdays", "weekends", "holidays"]]
