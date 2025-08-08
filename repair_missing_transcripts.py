import time
import asyncio
from datetime import datetime

from opensearch_client import get_opensearch_client, index_document  # ✅ Your existing OpenSearch logic
from embedder import EMBEDDER_AVAILABLE  # We won't embed here, just re-index
from app import fetch_evaluation_by_id  # ✅ Your FastAPI-compatible fetch
from embedder import process_evaluation  # ✅ Your existing processing function

client = get_opensearch_client()

INDEX_PATTERN = "eval-template-*"
PAGE_SIZE = 1000

def log(msg):
    print(f"[{datetime.now().isoformat()}] {msg}")

def get_missing_docs():
    """
    Query OpenSearch for documents missing evaluation or transcript content.
    """
    query = {
        "query": {
            "bool": {
                "should": [
                    {"bool": {"must_not": {"exists": {"field": "transcript_text"}}}},
                    {"bool": {"must_not": {"exists": {"field": "evaluation_text"}}}}
                ],
                "minimum_should_match": 1
            }
        },
        "_source": ["evaluationId", "template_id"],
        "size": PAGE_SIZE
    }

    response = client.search(index=INDEX_PATTERN, body=query)
    hits = response.get("hits", {}).get("hits", [])

    docs = []
    for hit in hits:
        src = hit.get("_source", {})
        evaluation_id = src.get("evaluationId")
        template_id = src.get("template_id")
        if evaluation_id and template_id:
            docs.append({"evaluationId": evaluation_id, "template_id": template_id})

    return docs


async def repair_all_missing():
    missing_docs = get_missing_docs()
    log(f"📦 Found {len(missing_docs)} evaluations missing transcript or evaluation text")

    repaired = 0
    skipped = 0
    failed = 0

    for doc in missing_docs:
        eval_id = doc["evaluationId"]
        try:
            evaluation = fetch_evaluation_by_id(eval_id)
            if not evaluation:
                log(f"⚠️ Skipping - API returned None for evaluation {eval_id}")
                skipped += 1
                continue

            result = await process_evaluation(evaluation)

            if result and result.get("status") == "success":
                repaired += 1
                log(f"✅ Repaired evaluation {eval_id}")
            else:
                failed += 1
                log(f"❌ Failed processing eval {eval_id}")

        except Exception as e:
            failed += 1
            log(f"❌ Error processing eval {eval_id}: {e}")

        await asyncio.sleep(0.05)

    log("\n📊 REPAIR SUMMARY")
    log(f"Repaired: {repaired}")
    log(f"Skipped: {skipped}")
    log(f"Failed:  {failed}")


if __name__ == "__main__":
    asyncio.run(repair_all_missing())
