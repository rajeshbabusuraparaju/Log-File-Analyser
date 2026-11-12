import pandas as pd
from google.cloud import aiplatform

# ==========================================
# 1. CONFIGURATION
# ==========================================
PROJECT_ID = "your-gcp-project-id"
LOCATION = "us-central1"
ENDPOINT_ID = "your-tabfm-endpoint-id"  # Vertex AI TabFM deployed endpoint ID

INPUT_EXCEL_PATH = "logs.xlsx"
OUTPUT_EXCEL_PATH = "classified_logs.xlsx"
SHEET_NAME = "Logs"  # Name of the worksheet tab inside your Excel file


# ==========================================
# 2. LOAD LOGS FROM LOCAL EXCEL FILE
# ==========================================
def load_logs_from_excel(file_path: str, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
    # Convert timestamps or non-string fields to string so JSON serialization succeeds
    for col in ["timestamp", "logger_name", "thread", "message"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


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

    # Prepare target rows with empty label column
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
# 5. SAVE RESULTS TO LOCAL EXCEL FILE
# ==========================================
def save_results_to_excel(df_results: pd.DataFrame, output_path: str, sheet_name: str):
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_results.to_excel(writer, sheet_name=sheet_name, index=False)


if __name__ == "__main__":
    # 1. Authenticate locally before running:
    #    gcloud auth application-default login

    # 2. Read logs from local Excel file
    logs_df = load_logs_from_excel(INPUT_EXCEL_PATH, SHEET_NAME)

    # 3. Classify against Log4j INFO standards via TabFM
    classified_df = classify_logs_with_tabfm(logs_df)

    # 4. Write results to a new local Excel file
    save_results_to_excel(classified_df, OUTPUT_EXCEL_PATH, SHEET_NAME)
    print(
        f"Successfully classified {len(classified_df)} log entries into '{OUTPUT_EXCEL_PATH}'."
    )
