import zipfile
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

# Set the baby birth date
BIRTHDATE = pd.to_datetime("2024-03-30 23:30:00.000000+02:00")

# Exclude data before the 6th of May for the baby sleping data
FIRST_DAY_SLEEP_DATA = pd.to_datetime("2024-05-06T00:00:00.000+02:00")

# Exclude data before the 6th of May for the feeding data
FIRST_DAY_FEEDING_DATA =pd.to_datetime("2024-05-06T00:00:00.000+02:00")


# Load data
def load_data(path: Path, json_file_name:str) -> list[dict]:
    with zipfile.ZipFile(path, "r") as zip_ref:
        with zip_ref.open(json_file_name) as json_file:
            data = json.load(json_file)
    return data


# Calculate the difference between birthdate and given date
def calculate_months_days(given_date:pd.Timestamp, birthdate:pd.Timestamp) -> tuple[int, int]:
    delta = relativedelta(given_date, birthdate)
    return delta.years * 12 + delta.months, delta.days


# Preprocess data for sleeping patterns
def preprocess_sleep(data: list[dict], birthdate:pd.Timestamp, child:bool) -> pd.DataFrame:
    df = pd.DataFrame(data)
    # Duration
    df["duration_seconds"] = df["duration"]
    df["duration_minutes"] = round(df["duration"] / 60)
    df["duration_hours"] = round((df["duration"] / 60) / 60, 1)
        
    # Extract time info for startDate
    df["start_datetime"] = pd.to_datetime(df["startDate"])
    df["start_day"] = df["start_datetime"].dt.day
    df["start_month"] = df["start_datetime"].dt.month
    df["start_year"] = df["start_datetime"].dt.year
    df["start_hour"] = df["start_datetime"].dt.hour
    df["start_minute"] = df["start_datetime"].dt.minute
    df["start_time"] = df["start_datetime"].dt.strftime("%H:%M")
    df["start_weekday"] = df["start_datetime"].dt.dayofweek + 1 # add 1 because: Monday=0, Sunday=6
    df["start_weekday_name"] = df["start_datetime"].dt.day_name()
    
    # Extract time info for endDate
    df["end_datetime"] = pd.to_datetime(df["endDate"])
    df["end_day"] = df["end_datetime"].dt.day
    df["end_month"] = df["end_datetime"].dt.month
    df["end_year"] = df["end_datetime"].dt.year
    df["end_hour"] = df["end_datetime"].dt.hour
    df["end_minute"] = df["end_datetime"].dt.minute
    df["end_time"] = df["end_datetime"].dt.strftime("%H:%M")
    df["end_weekday"] = df["end_datetime"].dt.dayofweek + 1 # add 1 because: Monday=0, Sunday=6
    df["end_weekday_name"] = df["end_datetime"].dt.day_name()

    # Add "night"
    # The night will be coded as the day where the night starts, for example:
    # 16 June from 22:30 to 23:45 is coded as night=16 June
    # 16 June from 22:30 to 01:45 is coded as night=16 June
    # 17 June from 03:30 to 05:45 is coded as night=16 June
    # 01 June from 01:30 to 02:30 is coded as night=31 May
    df["night_datetime"] = df["start_datetime"]
    # Code the date based on the hour
    if child:
        df["night_datetime"] = df["start_datetime"]
    else:
        df.loc[df["start_datetime"].dt.hour < 9, "night_datetime"] = df["start_datetime"] - pd.Timedelta(days=1)
        df.loc[df["start_datetime"].dt.hour >= 18, "night_datetime"] = df["start_datetime"]
    df["night"] = df["night_datetime"].dt.date
    df["night_day"] = df["night_datetime"].dt.day
    df["night_month"] = df["night_datetime"].dt.month
    df["night_year"] = df["night_datetime"].dt.year
    df["night_weekday"] = df["night_datetime"].dt.dayofweek + 1 # add 1 because: Monday=0, Sunday=6
    df["night_weekday_name"] = df["night_datetime"].dt.day_name()

    # Add months and days passed
    df[["months_passed", "days_passed"]] = [calculate_months_days(days, birthdate) for days in df["night_datetime"]]

    # Select relevant columns
    columns_to_keep = [
        "duration_seconds", "duration_minutes", "duration_hours", "endDate", "startDate",
        "start_datetime", "start_day", "start_month", "start_year", "start_hour",
        "start_minute", "start_time", "start_weekday", "start_weekday_name",
        "end_datetime", "end_day", "end_month", "end_year", "end_hour",
        "end_minute", "end_time", "end_weekday", "end_weekday_name",
        "night_datetime", "night", "night_day", "night_month", "night_year", "night_weekday", "night_weekday_name",
        "months_passed", "days_passed"
        ]
    
    return df[columns_to_keep]


def group_sleep(data: pd.DataFrame) -> pd.DataFrame:
    sleep_grouped = (
        data
        .groupby("night")
        .agg({
            "duration_seconds":"sum",
            "duration_minutes":"sum",
            "duration_hours":"sum",
            "night_datetime":"first",
            "night_day":"first",
            "night_month":"first",
            "night_year":"first",
            "night_weekday":"first",
            "night_weekday_name":"first",
            "months_passed":"first",
            "days_passed":"first"
        })
        .reset_index()
    )
    return sleep_grouped


def preprocess_feeding(data: list[dict], birthdate:pd.Timestamp) -> pd.DataFrame:
    df = pd.DataFrame(data)
    # Duration
    df["left_duration_seconds"] = df["leftDuration"]
    df["left_duration_minutes"] = round(df["leftDuration"] / 60)
    df["left_duration_hours"] = round((df["leftDuration"] / 60) / 60, 1)
    df["right_duration_seconds"] = df["rightDuration"]
    df["right_duration_minutes"] = round(df["rightDuration"] / 60)
    df["right_duration_hours"] = round((df["rightDuration"] / 60) / 60, 1)
        
    # Extract time info for startDate
    df["start_datetime"] = pd.to_datetime(df["startDate"])
    df["start_day"] = df["start_datetime"].dt.day
    df["start_month"] = df["start_datetime"].dt.month
    df["start_year"] = df["start_datetime"].dt.year
    df["start_hour"] = df["start_datetime"].dt.hour
    df["start_minute"] = df["start_datetime"].dt.minute
    df["start_time"] = df["start_datetime"].dt.strftime("%H:%M")
    df["start_weekday"] = df["start_datetime"].dt.dayofweek + 1 # add 1 because: Monday=0, Sunday=6
    df["start_weekday_name"] = df["start_datetime"].dt.day_name()
    
    # Extract time info for endDate
    df["end_datetime"] = pd.to_datetime(df["endDate"])
    df["end_day"] = df["end_datetime"].dt.day
    df["end_month"] = df["end_datetime"].dt.month
    df["end_year"] = df["end_datetime"].dt.year
    df["end_hour"] = df["end_datetime"].dt.hour
    df["end_minute"] = df["end_datetime"].dt.minute
    df["end_time"] = df["end_datetime"].dt.strftime("%H:%M")
    df["end_weekday"] = df["end_datetime"].dt.dayofweek + 1 # add 1 because: Monday=0, Sunday=6
    df["end_weekday_name"] = df["end_datetime"].dt.day_name()

    # day
    df["day"] = df["start_datetime"].dt.date

    # Add months and days passed
    df[["months_passed", "days_passed"]] = [calculate_months_days(days, birthdate) for days in df["start_datetime"]]

    # Select relevant columns
    columns_to_keep = [
        "left_duration_seconds", "left_duration_minutes", "left_duration_hours",
        "right_duration_seconds", "right_duration_minutes", "right_duration_hours",
        "endDate", "startDate",
        "start_datetime", "start_day", "start_month", "start_year", "start_hour",
        "start_minute", "start_time", "start_weekday", "start_weekday_name",
        "end_datetime", "end_day", "end_month", "end_year", "end_hour",
        "end_minute", "end_time", "end_weekday", "end_weekday_name",
        "day", "months_passed", "days_passed"
        ]
    
    return df[columns_to_keep]


def group_feeding(data: pd.DataFrame) -> pd.DataFrame:
    feeding_grouped = (
        data
        .groupby("day")
        .agg(
            left_duration_seconds=pd.NamedAgg(column="left_duration_seconds", aggfunc="sum"),
            left_duration_minutes=pd.NamedAgg(column="left_duration_minutes", aggfunc="sum"),
            left_duration_hours=pd.NamedAgg(column="left_duration_hours", aggfunc="sum"),
            right_duration_seconds=pd.NamedAgg(column="right_duration_seconds", aggfunc="sum"),
            right_duration_minutes=pd.NamedAgg(column="right_duration_minutes", aggfunc="sum"),
            right_duration_hours=pd.NamedAgg(column="right_duration_hours", aggfunc="sum"),
            start_datetime=pd.NamedAgg(column="start_datetime", aggfunc="first"),
            start_day=pd.NamedAgg(column="start_day", aggfunc="first"),
            start_month=pd.NamedAgg(column="start_month", aggfunc="first"),
            start_year=pd.NamedAgg(column="start_year", aggfunc="first"),
            start_weekday=pd.NamedAgg(column="start_weekday", aggfunc="first"),
            start_weekday_name=pd.NamedAgg(column="start_weekday_name", aggfunc="first"),
            months_passed=pd.NamedAgg(column="months_passed", aggfunc="first"),
            days_passed=pd.NamedAgg(column="days_passed", aggfunc="first"),
            count=pd.NamedAgg(column="start_day", aggfunc="count"),
        )
        .reset_index()
    )
    return feeding_grouped


if __name__ == "__main__":

    # Path to the zip files
    path_zip_my_data = Path("data", "my__babyplus_data.zip")
    path_zip_her_data = Path("data", "her__babyplus_data.zip")

    # Name of the JSON file inside the zip archive
    json_file_name = "babyplus_data_export.json"

    # Load data
    my_data = load_data(path_zip_my_data, json_file_name)
    her_data = load_data(path_zip_her_data, json_file_name)

    # Preprocessing sleeping data
    my_sleep_df = preprocess_sleep(my_data["baby_sleep"], BIRTHDATE, child=False)
    his_sleep_df = preprocess_sleep(her_data["baby_sleep"], BIRTHDATE, child=True)

    # Exclude data before FIRST_DAY_SLEEP_DATA
    his_sleep_df_filtered = his_sleep_df[his_sleep_df["start_datetime"] > FIRST_DAY_SLEEP_DATA]

    # Group by night
    my_sleep_grouped = group_sleep(my_sleep_df)
    his_sleep_grouped_filtered = group_sleep(his_sleep_df_filtered)

    # Preprocessing feeding data
    feeding_mydata = preprocess_feeding(my_data["baby_nursingfeed"], BIRTHDATE)
    feeding_herdata = preprocess_feeding(her_data["baby_nursingfeed"], BIRTHDATE)

    # Exclude data before the FIRST_DAY_FEEDING_DATA
    feeding_herdata_filtered = feeding_herdata[feeding_herdata["start_datetime"] > FIRST_DAY_FEEDING_DATA]
    
    # Concatenate feeding data
    feeding_combined = pd.concat(
        [feeding_mydata, feeding_herdata_filtered],
        ignore_index=True
    )

    # Group by day
    feeding_combine_grouped = group_feeding(feeding_combined)
    
    # Save data frame into csv files
    my_sleep_df.to_csv(Path("csv_files", "my_sleep_df.csv"), index=False)
    my_sleep_grouped.to_csv(Path("csv_files", "my_sleep_grouped.csv"), index=False)
    his_sleep_grouped_filtered.to_csv(Path("csv_files", "his_sleep_grouped_filtered.csv"), index=False)
    feeding_combine_grouped.to_csv(Path("csv_files", "feeding_df_combine_grouped.csv"), index=False)
