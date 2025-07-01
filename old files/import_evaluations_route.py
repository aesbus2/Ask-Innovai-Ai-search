# import_evaluations_route.py

from fastapi import APIRouter
from evaluation_import_job import import_evaluations_from_api
import asyncio

router = APIRouter()

@router.post("/import_evaluations")
async def run_eval_import():
    """
    Manually trigger evaluation import from Admin panel
    """
    try:
        await import_evaluations_from_api()
        return {"status": "success", "message": "Evaluation import completed"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
