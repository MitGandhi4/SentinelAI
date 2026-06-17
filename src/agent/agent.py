import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage

from src.enrichment.enrichment import load_attack_data, get_attack_enrichment, get_cve_details
from src.enrichment.rag_pipeline import search_techniques

# ── 1. Load ATT&CK data once at startup ───────────────────────────
print("Loading ATT&CK data...")
TECHNIQUES = load_attack_data("data/raw/enterprise-attack.json")
print(f"Loaded {len(TECHNIQUES)} techniques")


# ── 2. Define agent state ──────────────────────────────────────────
# This is what the agent tracks throughout its reasoning process
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


# ── 3. Define tools the agent can use ─────────────────────────────
@tool
def search_attack_techniques(query: str) -> str:
    """
    Searches the MITRE ATT&CK knowledge base for techniques
    relevant to a given query. Use this to find attack patterns
    that match observed behavior.
    """
    results = search_techniques(query, n_results=3)
    output = "Relevant MITRE ATT&CK Techniques:\n"
    for r in results:
        output += f"- {r['technique_id']}: {r['technique_name']} "
        output += f"(Tactics: {r['tactics']}, Relevance: {r['relevance_score']:.2f})\n"
        output += f"  URL: {r['mitre_url']}\n"
    return output


@tool
def get_technique_details(attack_label: str) -> str:
    """
    Gets detailed MITRE ATT&CK information for a specific attack label
    from the CICIDS dataset (e.g. 'PortScan', 'DDoS', 'SSH-Patator').
    """
    enrichment = get_attack_enrichment(attack_label, TECHNIQUES)
    output = f"ATT&CK Enrichment for '{attack_label}':\n"
    output += f"- Technique: {enrichment['technique_id']} — {enrichment['technique_name']}\n"
    output += f"- Tactics: {', '.join(enrichment['tactics'])}\n"
    output += f"- Description: {enrichment['description'][:300]}...\n"
    output += f"- MITRE URL: {enrichment['mitre_url']}\n"
    return output


@tool
def lookup_cve(cve_id: str) -> str:
    """
    Looks up detailed information about a CVE (Common Vulnerability
    and Exposure) from the NVD database. Use this when a specific
    CVE is suspected or known.
    """
    cve = get_cve_details(cve_id)
    if "error" in cve:
        return f"Error: {cve['error']}"
    output = f"CVE Details for {cve['cve_id']}:\n"
    output += f"- Severity: {cve['severity']} (Score: {cve['score']})\n"
    output += f"- Description: {cve['description']}\n"
    output += f"- NVD URL: {cve['nvd_url']}\n"
    return output


@tool
def analyze_network_flow(flow_features: str) -> str:
    """
    Analyzes network flow features to identify suspicious patterns.
    Input should be a comma-separated list of key:value pairs
    e.g. 'flow_duration:1000, total_fwd_packets:500, flow_bytes_per_sec:9000000'
    """
    output = "Network Flow Analysis:\n"
    features = {}
    for item in flow_features.split(","):
        if ":" in item:
            key, val = item.strip().split(":", 1)
            features[key.strip()] = val.strip()

    for key, val in features.items():
        try:
            num_val = float(val)
            if "duration" in key.lower() and num_val > 1000000:
                output += f"⚠ High flow duration ({num_val}) — possible slow attack\n"
            if "packets" in key.lower() and num_val > 10000:
                output += f"⚠ High packet count ({num_val}) — possible DoS/DDoS\n"
            if "bytes" in key.lower() and num_val > 1000000:
                output += f"⚠ High byte rate ({num_val}) — possible data exfiltration or flood\n"
        except:
            pass

    if not any(k in output for k in ["⚠"]):
        output += "No immediately suspicious patterns detected in provided features.\n"

    return output


# ── 4. Initialize LLM and bind tools ──────────────────────────────
llm = ChatOllama(model="llama3.2", temperature=0)
tools = [search_attack_techniques, get_technique_details, lookup_cve, analyze_network_flow]
llm_with_tools = llm.bind_tools(tools)


# ── 5. Define agent node ───────────────────────────────────────────
def agent_node(state: AgentState) -> AgentState:
    """
    The agent node — LLM thinks and decides what to do next.
    Either calls a tool or produces a final answer.
    """
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ── 6. Build the LangGraph state machine ──────────────────────────
def build_agent():
    # Create the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    # Set entry point
    graph.set_entry_point("agent")

    # Add conditional edge — if agent wants to use a tool go to tools
    # otherwise end
    graph.add_conditional_edges(
        "agent",
        tools_condition  # built-in condition that checks for tool calls
    )

    # After tools always go back to agent to continue reasoning
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── 7. Main analysis function ──────────────────────────────────────
def analyze_alert(
        attack_label: str,
        anomaly_score: float,
        flow_features: dict
) -> str:
    """
    Main entry point for the agent. Given an alert from our ML models,
    the agent reasons about what happened and produces an analysis.
    """
    agent = build_agent()
    feature_str = ", ".join([f"{k}:{v}" for k, v in flow_features.items()])
    prompt = f"""An anomaly was detected: {attack_label} attack with anomaly score {anomaly_score:.4f}.
    Network flow features: {feature_str}

    Use get_technique_details to look up this attack type. If this is Heartbleed, also use lookup_cve with CVE-2014-0160. Then provide a brief security analysis."""
    # Run the agent
    result = agent.invoke({
        "messages": [HumanMessage(content=prompt)]
    })

    # Extract the final response
    final_message = result["messages"][-1].content
    return final_message

# ── 8. Test the agent ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SentinelAI Agent — Test Run")
    print("=" * 60 + "\n")

    # Simulate an alert from our ML ensemble
    test_alert = {
        "attack_label": "Heartbleed",
        "anomaly_score": 0.95,
        "flow_features": {
            "flow_duration": 120000,
            "total_fwd_packets": 50,
            "total_bwd_packets": 50,
            "flow_bytes_per_sec": 5000,
            "destination_port": 443
        }
    }

    print(f"Alert: {test_alert['attack_label']} detected")
    print(f"Anomaly Score: {test_alert['anomaly_score']}")
    print("\nRunning agent analysis...\n")

    analysis = analyze_alert(
        attack_label=test_alert["attack_label"],
        anomaly_score=test_alert["anomaly_score"],
        flow_features=test_alert["flow_features"]
    )

    print("Agent Analysis:")
    print("-" * 40)
    print(analysis)