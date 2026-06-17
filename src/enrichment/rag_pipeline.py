import chromadb
from sentence_transformers import SentenceTransformer
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.enrichment.enrichment import load_attack_data

# ── 1. Load full ATT&CK dataset ────────────────────────────────────
print("Loading full ATT&CK dataset...")
techniques = load_attack_data("data/raw/enterprise-attack.json")
print(f"Loaded {len(techniques)} techniques")

# ── 2. Initialize ChromaDB ─────────────────────────────────────────
# ChromaDB stores our vectors on disk so we don't rebuild every time
client = chromadb.PersistentClient(path="data/processed/chromadb")
collection_name = "mitre_attack"

# Load existing collection if it exists, otherwise create a new one
# This avoids re-embedding all 697 techniques every single run
existing = [c.name for c in client.list_collections()]
if collection_name in existing:
    collection = client.get_collection(name=collection_name)
    print(f"Loaded existing ChromaDB collection with {collection.count()} techniques")
else:
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # use cosine similarity for text
    )
    print("Created new ChromaDB collection")

# ── 3. Load embedding model ────────────────────────────────────────
# all-MiniLM-L6-v2 is a fast, lightweight embedding model
# It converts text to 384-dimensional vectors
print("\nLoading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded")

# ── 4. Embed and store all techniques (only if not already done) ─
if collection.count() == 0:
    print("\nEmbedding and storing all ATT&CK techniques...")
    print("This will take a few minutes...")

    # Process in batches of 100 for efficiency
    technique_list = list(techniques.values())
    batch_size = 100

    for i in range(0, len(technique_list), batch_size):
        batch = technique_list[i:i + batch_size]

        # Create the text we'll embed — combine name, tactics, and description
        texts = [
            f"{t['technique_name']} ({t['technique_id']}). "
            f"Tactics: {', '.join(t['tactics'])}. "
            f"{t['description'][:500]}"
            for t in batch
        ]

        # Generate embeddings
        embeddings = embedder.encode(texts).tolist()

        # Store in ChromaDB
        collection.add(
            ids=[t["technique_id"] for t in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{
                "technique_id": t["technique_id"],
                "technique_name": t["technique_name"],
                "tactics": ", ".join(t["tactics"]),
                "mitre_url": t["mitre_url"]
            } for t in batch]
        )

        print(f"Stored techniques {i + 1} to {min(i + batch_size, len(technique_list))}")

    print(f"\nAll {len(technique_list)} techniques stored in ChromaDB")
else:
    print(f"Using existing embeddings ({collection.count()} techniques) — skipping re-embedding")


# ── 5. Semantic search function ────────────────────────────────────
def search_techniques(query: str, n_results: int = 3) -> list:
    """
    Searches the vector store for techniques relevant to a query.
    Returns the top n_results most semantically similar techniques.
    """
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    matches = []
    for i in range(len(results["ids"][0])):
        matches.append({
            "technique_id": results["metadatas"][0][i]["technique_id"],
            "technique_name": results["metadatas"][0][i]["technique_name"],
            "tactics": results["metadatas"][0][i]["tactics"],
            "mitre_url": results["metadatas"][0][i]["mitre_url"],
            "relevance_score": 1 - results["distances"][0][i]
        })
    return matches


if __name__ == "__main__":
    print("\n=== Semantic Search Tests ===\n")

    test_queries = [
        "slow HTTP connections exhausting server resources",
        "scanning network for open ports and services",
        "stealing credentials through password guessing",
        "attacker controlling compromised machines remotely"
    ]

    for query in test_queries:
        print(f"Query: '{query}'")
        results = search_techniques(query, n_results=2)
        for r in results:
            print(f"  → {r['technique_id']} — {r['technique_name']} "
                  f"(Tactics: {r['tactics']}, Score: {r['relevance_score']:.3f})")
        print()