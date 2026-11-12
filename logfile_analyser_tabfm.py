import gspread
import pandas as pd
from google.cloud import aiplatform
from google.oauth2.service_account import Credentials

# ==========================================
# 1. CONFIGURATION
# ==========================================
PROJECT_ID = "your-gcp-project-id"
LOCATION = "us-central1"
ENDPOINT_ID = "your-tabfm-endpoint-id"  # Vertex AI TabFM deployed endpoint ID

SERVICE_ACCOUNT_FILE = "path/to/service-account.json"
SHEET_ID = "your-google-sheet-id"
WORKSHEET_NAME = "Logs"


# ==========================================
# 2. LOAD LOGS FROM GOOGLE SHEETS
# ==========================================
def load_logs_from_sheets(sheet_id: str, worksheet_name: str) -> pd.DataFrame:
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scopes)
    client = gspread.authorize(creds)

    sheet = client.open_by_key(sheet_id).worksheet(worksheet_name)
    records = sheet.get_all_records()
    return pd.DataFrame(records)


# ==========================================
# 3. BUILD FEW-SHOT CONTEXT FOR LOG4J INFO
# ==========================================
def get_log4j_reference_examples() -> pd.DataFrame:
    """
    Provides few-shot tabular context defining Log4j INFO compliance
    for TabFM in-context learning.
    """
    examples = [
        {
            "timestamp": "2026-11-12 08:00:01",
            "logger_name": "com.app.ServerMain",
            "thread": "main",
            "message": "Application started successfully on port 8080",
            "meets_info_guidelines": "COMPLIANT",
        },
        {
            "timestamp": "2026-11-12 08:01:15",
            "logger_name": "com.app.db.PoolManager",
            "thread": "worker-1",
            "message": "Connection pool initialized with 20 active connections",
            "meets_info_guidelines": "COMPLIANT",
        },
        {
            "timestamp": "2026-11-12 08:02:10",
            "logger_name": "com.app.auth.TokenParser",
            "thread": "http-nio-8080-exec-2",
            "message": "Inspecting JWT header payload: {alg: HS256, typ: JWT}, user_id=49102",
            "meets_info_guidelines": "NON_COMPLIANT_DEBUG",
        },
        {
            "timestamp": "2026-11-12 08:03:44",
            "logger_name": "com.app.payment.Gateway",
            "thread": "http-nio-8080-exec-5",
            "message": "NullPointerException at PaymentProcessor.java:142 - transaction aborted",
            "meets_info_guidelines": "NON_COMPLIANT_ERROR",
        },
        {
            "timestamp": "2026-11-12 08:04:12",
            "logger_name": "com.app.cache.RedisClient",
            "thread": "scheduler-1",
            "message": "Cache latency exceeded 250ms threshold; retrying request",
            "meets_info_guidelines": "NON_COMPLIANT_WARN",
        },
    ]
    return pd.DataFrame(examples)


# ==========================================
# 4. CLASSIFY USING VERTEX AI TABFM
# ==========================================
def classify_logs_with_tabfm(df_logs: pd.DataFrame) -> pd.DataFrame:
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)

    context_df = get_log4j_reference_examples()
    target_column = "meets_info_guidelines"

    # Ensure target column exists as None/empty in target rows
    df_target = df_logs.copy()
    df_target[target_column] = None

    # Format payload for TabFM in-context tabular prediction
    instances = [
        {
            "context_rows": context_df.to_dict(orient="records"),
            "target_rows": df_target[
                ["timestamp", "logger_name", "thread", "message", target_column]
            ].to_dict(orient="records"),
            "target_column": target_column,
            "task_type": "classification",
        }
    ]

    response = endpoint.predict(instances=instances)

    # Extract predicted labels and confidence scores
    predictions = response.predictions[0]
    df_logs["predicted_classification"] = [p["label"] for p in predictions]
    df_logs["confidence"] = [round(p["score"], 4) for p in predictions]

    return df_logs


# ==========================================
# 5. WRITE RESULTS BACK TO GOOGLE SHEETS
# ==========================================
def save_results_to_sheets(
    df_results: pd.DataFrame, sheet_id: str, worksheet_name: str
):
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scopes)
    client = gspread.authorize(creds)

    sheet = client.open_by_key(sheet_id).worksheet(worksheet_name)
    sheet.clear()
    sheet.update([df_results.columns.values.tolist()] + df_results.values.tolist())


if __name__ == "__main__":
    # 1. Read logs (Expected columns: timestamp, logger_name, thread, message)
    logs_df = load_logs_from_sheets(SHEET_ID, WORKSHEET_NAME)

    # 2. Classify against Log4j INFO standards via TabFM
    classified_df = classify_logs_with_tabfm(logs_df)

    # 3. Write classifications back to Google Sheets
    save_results_to_sheets(classified_df, SHEET_ID, WORKSHEET_NAME)
    print(f"Successfully classified {len(classified_df)} log entries.")
