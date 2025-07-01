# evaluation_import_job.py

import os
import requests
import asyncio
from typing import List
from eval_indexing_logic import process_evaluation_document

# Example config from environment or constants
EVALUATION_API_URL = os.getenv("EVALUATION_API_URL", "https://your.api/evaluations")
AUTH_HEADER = {"Authorization": os.getenv("API_AUTH", "Bearer your_token")}

async def import_evaluations_from_api():
    try:
        response = requests.get(EVALUATION_API_URL, headers=AUTH_HEADER, timeout=60)
        response.raise_for_status()
        data = response.json()

        if not data or "evaluations" not in data:
            print("No evaluations found in API response.")
            return

        documents: List[dict] = data["evaluations"]
        print(f"Found {len(documents)} evaluations to index")

        for doc in documents:
            await process_evaluation_document(doc)

        print("Evaluation import completed.")

    except Exception as e:
        print(f"Evaluation import failed: {e}")

# For manual local testing
if __name__ == "__main__":
    asyncio.run(import_evaluations_from_api())
