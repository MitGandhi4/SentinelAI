import requests
import json


# ── 1. Load full MITRE ATT&CK dataset ─────────────────────────────
def load_attack_data(filepath: str) -> dict:
    """
    Loads the full MITRE ATT&CK enterprise dataset from the downloaded JSON file.
    Returns a dictionary mapping technique IDs to their full details.
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    techniques = {}
    for obj in data["objects"]:
        # We only want attack-pattern objects (techniques)
        if obj["type"] != "attack-pattern":
            continue
        # Skip deprecated or revoked techniques
        if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
            continue

        # Extract technique ID (e.g. T1046)
        technique_id = None
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                technique_id = ref.get("external_id")
                break

        if not technique_id:
            continue

        # Extract tactic names
        tactics = [
            phase["phase_name"].replace("-", " ").title()
            for phase in obj.get("kill_chain_phases", [])
            if phase["kill_chain_name"] == "mitre-attack"
        ]

        techniques[technique_id] = {
            "technique_id": technique_id,
            "technique_name": obj.get("name", "Unknown"),
            "description": obj.get("description", "No description available"),
            "tactics": tactics,
            "mitre_url": f"https://attack.mitre.org/techniques/{technique_id}/"
        }

    print(f"Loaded {len(techniques)} ATT&CK techniques")
    return techniques


# ── 2. Map our dataset labels to ATT&CK technique IDs ─────────────
LABEL_TO_TECHNIQUE_ID = {
    "DDoS": "T1498",
    "DoS Hulk": "T1499",
    "DoS GoldenEye": "T1499",
    "DoS slowloris": "T1499",
    "DoS Slowhttptest": "T1499",
    "PortScan": "T1046",
    "FTP-Patator": "T1110",
    "SSH-Patator": "T1110",
    "Bot": "T1071",
    "Web Attack Brute Force": "T1110",
    "Web Attack XSS": "T1059",
    "Web Attack Sql Injection": "T1190",
    "Infiltration": "T1200",
    "Heartbleed": "T1190"
}


def get_attack_enrichment(label: str, techniques: dict) -> dict:
    """
    Given an attack label and the full techniques dictionary,
    returns the full MITRE ATT&CK enrichment for that label.
    """
    clean_label = label.replace("Web Attack \u00ef\u00bf\u00bd ", "Web Attack ")
    technique_id = LABEL_TO_TECHNIQUE_ID.get(clean_label)

    if technique_id and technique_id in techniques:
        return techniques[technique_id]

    return {
        "technique_id": "Unknown",
        "technique_name": "Unknown",
        "description": "No MITRE ATT&CK mapping found",
        "tactics": [],
        "mitre_url": "https://attack.mitre.org/"
    }


# ── 3. CVE lookup using NVD API ────────────────────────────────────
def get_cve_details(cve_id: str) -> dict:
    """
    Looks up a CVE ID using the NVD public API.
    """
    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if data["totalResults"] == 0:
            return {"error": f"CVE {cve_id} not found"}

        cve = data["vulnerabilities"][0]["cve"]

        severity = "Unknown"
        score = "Unknown"
        if "metrics" in cve:
            if "cvssMetricV31" in cve["metrics"]:
                cvss = cve["metrics"]["cvssMetricV31"][0]["cvssData"]
                severity = cvss["baseSeverity"]
                score = cvss["baseScore"]
            elif "cvssMetricV2" in cve["metrics"]:
                cvss = cve["metrics"]["cvssMetricV2"][0]["cvssData"]
                score = cvss["baseScore"]
                severity = "See NVD for severity"

        description = "No description available"
        for desc in cve["descriptions"]:
            if desc["lang"] == "en":
                description = desc["value"]
                break

        return {
            "cve_id": cve_id,
            "severity": severity,
            "score": score,
            "description": description[:300] + "..." if len(description) > 300 else description,
            "nvd_url": f"https://nvd.nist.gov/vuln/detail/{cve_id}"
        }

    except Exception as e:
        return {"error": f"Failed to fetch CVE: {str(e)}"}


# ── 4. Known CVE mappings ──────────────────────────────────────────
ATTACK_CVE_MAP = {
    "Heartbleed": "CVE-2014-0160",
    "Web Attack Sql Injection": None,
    "Web Attack XSS": None,
}


def get_cve_for_attack(label: str) -> dict | None:
    clean_label = label.replace("Web Attack \u00ef\u00bf\u00bd ", "Web Attack ")
    cve_id = ATTACK_CVE_MAP.get(clean_label)
    if cve_id:
        return get_cve_details(cve_id)
    return None


# ── 5. Test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    techniques = load_attack_data("data/raw/enterprise-attack.json")

    print("\n=== MITRE ATT&CK Enrichment (Full Dataset) ===\n")
    test_labels = ["PortScan", "DDoS", "SSH-Patator", "Heartbleed"]

    for label in test_labels:
        enrichment = get_attack_enrichment(label, techniques)
        print(f"Attack: {label}")
        print(f"  Technique: {enrichment['technique_id']} — {enrichment['technique_name']}")
        print(f"  Tactics: {', '.join(enrichment['tactics'])}")
        print(f"  Description: {enrichment['description'][:150]}...")
        print()

    print("=== CVE Lookup ===\n")
    cve = get_cve_details("CVE-2014-0160")
    print(f"CVE: {cve.get('cve_id')} | Severity: {cve.get('severity')} | Score: {cve.get('score')}")
    print(f"Description: {cve.get('description')}")