import argparse
import csv
import json
import os
import sys

# from collections import defaultdict

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from openai import OpenAI


@dataclass
class LogTemplateStats:
    template_mined: str
    first_line_num: int
    count: int
    sample_line: str
    recommendation: str = ""
    reasoning: str = ""


# Dummy comment

# Dummy comment 2 in the feature branch to try PRs


def parse_log_with_drain(file_path: str) -> Dict[str, LogTemplateStats]:
    """Streams log file line-by-line using Drain3 to cluster templates and capture stats."""
    config = TemplateMinerConfig()
    # Configure sensible defaults for masking numeric/token noise
    config.profiling_enabled = False
    miner = TemplateMiner(config=config)

    # Key: template_mined string -> LogTemplateStats
    template_registry: Dict[str, LogTemplateStats] = {}

    print(f"[*] Parsing log file: {file_path}")
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            result = miner.add_log_message(line)
            template = result["template_mined"]

            if template not in template_registry:
                template_registry[template] = LogTemplateStats(
                    template_mined=template,
                    first_line_num=line_num,
                    count=1,
                    sample_line=line,
                )
            else:
                template_registry[template].count += 1

    print(
        f"[*] Finished parsing. Identified {len(template_registry)} unique template clusters."
    )
    return template_registry


def classify_templates_with_llm(
    file_name: str,
    templates: Dict[str, LogTemplateStats],
    client: OpenAI,
    model: str = "gpt-4o-mini",
    batch_size: int = 10,
) -> List[LogTemplateStats]:
    """Batches unique log templates and queries an LLM to evaluate signal-to-noise value."""
    template_list = list(templates.values())
    total_templates = len(template_list)
    classified_templates: List[LogTemplateStats] = []

    system_prompt = (
        "You are an expert site reliability and software observability engineer.\n"
        "Analyze clusters of application log templates and determine if each is 'Needed' or 'Not needed'.\n\n"
        "Classification criteria:\n"
        "- Needed: Errors, warnings, authentication events, state mutations, transactional boundaries, audit trails, security warnings.\n"
        "- Not needed: High-frequency polling loops, routine heartbeat/health checks, verbose trace/debug dumps, progress meters, redundant operational pings.\n\n"
        "Return a valid JSON object with a single key 'evaluations', containing an array of objects matching the input order:\n"
        '{\n  "evaluations": [\n    {"id": <int>, "recommendation": "Needed" | "Not needed", "reasoning": "<concise justification>"}\n  ]\n}'
    )

    for i in range(0, total_templates, batch_size):
        batch = template_list[i : i + batch_size]
        print(
            f"[*] Classifying batch {i // batch_size + 1}/{(total_templates + batch_size - 1) // batch_size}..."
        )

        user_content = {"file_name": file_name, "candidates": []}

        for idx, item in enumerate(batch):
            user_content["candidates"].append(
                {
                    "id": idx,
                    "template": item.template_mined,
                    "frequency_count": item.count,
                    "first_seen_line": item.first_line_num,
                    "concrete_sample": item.sample_line,
                }
            )

        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_content)},
                ],
            )
            raw_json = response.choices[0].message.content
            parsed = json.loads(raw_json)
            evals = parsed.get("evaluations", [])

            for idx, item in enumerate(batch):
                match = next((e for e in evals if e.get("id") == idx), None)
                if match:
                    item.recommendation = match.get("recommendation", "Needed")
                    item.reasoning = match.get("reasoning", "No reason provided.")
                else:
                    item.recommendation = "Needed"
                    item.reasoning = "Defaulted due to response mismatch."
                classified_templates.append(item)

        except Exception as e:
            print(f"[!] Error in LLM call for batch: {e}")
            for item in batch:
                item.recommendation = "Needed"
                item.reasoning = f"Classification failed: {e}"
                classified_templates.append(item)

    # Sort descending by occurrence count to highlight biggest space-savers first
    classified_templates.sort(key=lambda x: x.count, reverse=True)
    return classified_templates


def export_to_csv(file_name: str, records: List[LogTemplateStats], output_path: str):
    """Exports findings to CSV adhering to the specified 6-column format."""
    headers = [
        "File Name",
        "Line number",
        "Log entry format",
        "No of times the entry has been found in the log file",
        "Recommendation. Needed or not needed.",
        "Reasoning justifying the recommendation.",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in records:
            writer.writerow(
                [
                    file_name,
                    r.first_line_num,
                    r.template_mined,
                    r.count,
                    r.recommendation,
                    r.reasoning,
                ]
            )
    print(f"[✔] CSV export saved to: {output_path}")


def export_to_markdown(
    file_name: str, records: List[LogTemplateStats], output_path: str
):
    """Exports findings to Markdown adhering to the specified 6-column format."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Log Analysis Classification Report: `{file_name}`\n\n")
        f.write(
            "| File Name | Line number | Log entry format | Count | Recommendation | Reasoning |\n"
        )
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in records:
            clean_template = r.template_mined.replace("|", "\\|")
            clean_reason = r.reasoning.replace("|", "\\|")
            f.write(
                f"| {file_name} | {r.first_line_num} | `{clean_template}` | {r.count} | **{r.recommendation}** | {clean_reason} |\n"
            )
    print(f"[✔] Markdown export saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cluster logs via Drain3 and classify value via LLM."
    )
    parser.add_argument("log_file", type=str, help="Path to raw application log file")
    parser.add_argument(
        "--format",
        choices=["csv", "md", "both"],
        default="both",
        help="Export format (csv, md, or both)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="log_classification_report",
        help="Output filename prefix without extension",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model identifier",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        sys.exit(f"[!] Error: File '{args.log_file}' not found.")

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("[!] Error: OPENAI_API_KEY environment variable is not set.")

    client = OpenAI()

    # Step 1: Drain3 log template extraction & aggregation
    templates = parse_log_with_drain(str(log_path))

    # Step 2: LLM semantic evaluation (batched)
    classified_records = classify_templates_with_llm(
        file_name=log_path.name,
        templates=templates,
        client=client,
        model=args.model,
    )

    # Step 3: Export in requested format(s)
    if args.format in ("csv", "both"):
        export_to_csv(log_path.name, classified_records, f"{args.output_prefix}.csv")
    if args.format in ("md", "both"):
        export_to_markdown(
            log_path.name, classified_records, f"{args.output_prefix}.md"
        )


if __name__ == "__main__":
    main()
