import json
import re
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from src.enrichment.enrichment import load_attack_data, get_attack_enrichment, get_cve_details
from src.enrichment.rag_pipeline import search_techniques

# ── 1. Load ATT&CK data ────────────────────────────────────────────
print("Loading ATT&CK data for report generator...")
TECHNIQUES = load_attack_data("data/raw/enterprise-attack.json")

# ── 2. Initialize LLM (no tools needed here, just structured output) ─
llm = ChatOllama(model="llama3.2", temperature=0)

# ── 3. Define the report schema as a prompt template ──────────────
REPORT_SCHEMA_PROMPT = """You are a security incident report generator. 
Given the alert details below, respond with ONLY a valid JSON object — no other text before or after.

Alert Details:
- Attack Type: {attack_label}
- Anomaly Score: {anomaly_score}
- MITRE Technique: {technique_id} - {technique_name}
- Tactic: {tactic}
- Technique Description: {technique_description}

Respond with ONLY this exact JSON structure, filling in the values:
{{
  "incident_id": "auto-generated",
  "severity": "Critical, High, Medium, or Low based on the anomaly score and attack type",
  "attack_type": "{attack_label}",
  "mitre_technique": "{technique_id} - {technique_name}",
  "tactic": "{tactic}",
  "summary": "one sentence summary of what happened",
  "likely_attacker_goal": "what the attacker is trying to achieve",
  "affected_assets": "what system or service this likely targets based on the attack type",
  "recommended_actions": ["action 1", "action 2", "action 3"],
  "confidence": "High, Medium, or Low — your confidence in this assessment"
}}"""


# ── 4. Function to extract JSON from potentially messy LLM output ─
def extract_json(text: str) -> dict:
    """
    Tries to parse text as JSON directly. If that fails, searches
    for a JSON object embedded in surrounding text and extracts it.
    """
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find JSON object using regex
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # If all else fails, return an error structure
    return {
        "error": "Failed to parse structured output",
        "raw_output": text
    }


# ── 5. Main report generation function ────────────────────────────
def generate_incident_report(
        attack_label: str,
        anomaly_score: float
) -> dict:
    """
    Generates a structured incident report for a detected attack.
    Returns a dictionary matching our report schema.
    """
    # Get MITRE ATT&CK enrichment
    enrichment = get_attack_enrichment(attack_label, TECHNIQUES)

    # Build the prompt with real enrichment data filled in
    prompt = REPORT_SCHEMA_PROMPT.format(
        attack_label=attack_label,
        anomaly_score=anomaly_score,
        technique_id=enrichment["technique_id"],
        technique_name=enrichment["technique_name"],
        tactic=", ".join(enrichment["tactics"]),
        technique_description=enrichment["description"][:300]
    )

    # Call the LLM
    response = llm.invoke([HumanMessage(content=prompt)])

    # Parse the JSON response
    report = extract_json(response.content)

    # Add metadata we generate ourselves rather than trusting the LLM
    report["incident_id"] = f"INC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    report["timestamp"] = datetime.now().isoformat()
    report["anomaly_score"] = anomaly_score

    return report


# ── 6. Format the report as Markdown ───────────────────────────────
def format_report_as_markdown(report: dict) -> str:
    """
    Takes a structured report dictionary and formats it as a
    clean, readable Markdown incident report.
    """
    if "error" in report:
        return f"# Report Generation Failed\n\n{report.get('raw_output', 'Unknown error')}"

    actions = report.get("recommended_actions", [])
    actions_md = "\n".join([f"- {action}" for action in actions])

    markdown = f"""# Security Incident Report

**Incident ID:** {report.get('incident_id', 'N/A')}  
**Timestamp:** {report.get('timestamp', 'N/A')}  
**Severity:** {report.get('severity', 'Unknown')}  
**Confidence:** {report.get('confidence', 'Unknown')}

## Summary
{report.get('summary', 'No summary available')}

## Attack Details
- **Attack Type:** {report.get('attack_type', 'Unknown')}
- **MITRE ATT&CK Technique:** {report.get('mitre_technique', 'Unknown')}
- **Tactic:** {report.get('tactic', 'Unknown')}
- **Anomaly Score:** {report.get('anomaly_score', 'N/A')}

## Analysis
**Likely Attacker Goal:** {report.get('likely_attacker_goal', 'Unknown')}

**Affected Assets:** {report.get('affected_assets', 'Unknown')}

## Recommended Actions
{actions_md}

---
*Generated by SentinelAI*
"""
    return markdown


# ── 7. Test the full pipeline ──────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SentinelAI Incident Report Generator — Test")
    print("=" * 60 + "\n")

    test_cases = [
        {"attack_label": "PortScan", "anomaly_score": 0.72},
        {"attack_label": "Heartbleed", "anomaly_score": 0.95},
    ]

    for case in test_cases:
        print(f"Generating report for: {case['attack_label']}")
        report = generate_incident_report(
            attack_label=case["attack_label"],
            anomaly_score=case["anomaly_score"]
        )

        print("\nStructured JSON:")
        print(json.dumps(report, indent=2))

        markdown_report = format_report_as_markdown(report)
        print("\nMarkdown Report:")
        print(markdown_report)
        print("\n" + "-" * 60 + "\n")