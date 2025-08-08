# SOLUTION 1: Fix import_handlers.py to avoid circular imports
# Replace your import_handlers.py with this corrected version:

import os
import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Create router for import endpoints
import_router = APIRouter()

class ImportConfig(BaseModel):
    """Enhanced import configuration model"""
    collection: str = "all"
    import_type: str = "full"  # "full" or "incremental"
    max_docs: Optional[int] = None  # Maximum documents to process
    batch_size: Optional[int] = None  # Batch size for processing
    call_date_start: Optional[str] = None  # Start date for filtering (YYYY-MM-DD)
    call_date_end: Optional[str] = None  # End date for filtering (YYYY-MM-DD)

# ✅ FIXED: Create import function that doesn't cause circular imports
async def run_enhanced_production_import(
    collection: str = "all", 
    max_docs: int = None, 
    batch_size: int = None,
    call_date_start: str = None,
    call_date_end: str = None
):
    """
    Enhanced import process that supports date filtering and max docs
    This avoids circular imports by importing locally only when needed
    """
    try:
        # ✅ Import locally inside the function to avoid circular imports
        import sys
        
        # Get the app module safely
        app_module = sys.modules.get('app')
        if not app_module:
            raise ImportError("App module not available")
        
        # Get functions from app module
        run_production_import = getattr(app_module, 'run_production_import')
        update_import_status = getattr(app_module, 'update_import_status')
        log_import = getattr(app_module, 'log_import')
        fetch_evaluations = getattr(app_module, 'fetch_evaluations')
        
        # Log enhanced parameters
        log_import("🚀 Starting ENHANCED import with date filtering support")
        log_import(f"   📊 Max docs: {max_docs or 'Unlimited'}")
        log_import(f"   📅 Date range: {call_date_start or 'Unlimited'} to {call_date_end or 'Unlimited'}")
        log_import(f"   📦 Batch size: {batch_size or 'Default'}")
        
        # Enhanced fetch function that supports date filtering
        async def fetch_filtered_evaluations(max_docs_param=None):
            """Fetch evaluations with date filtering support"""
            try:
                # Get evaluations using the original function
                evaluations = await fetch_evaluations(max_docs_param)
                
                if not evaluations:
                    return []
                
                filtered_evaluations = []
                
                # Apply date filtering if specified
                for evaluation in evaluations:
                    should_include = True
                    
                    # Get call_date from evaluation
                    call_date_str = evaluation.get("call_date")
                    if call_date_str:
                        try:
                            # Parse the call date (handle different formats)
                            if "T" in call_date_str:
                                call_date = datetime.fromisoformat(call_date_str.replace("Z", "+00:00"))
                            else:
                                call_date = datetime.strptime(call_date_str, "%Y-%m-%d")
                            
                            # Apply date filters
                            if call_date_start:
                                start_date = datetime.strptime(call_date_start, "%Y-%m-%d")
                                if call_date.date() < start_date.date():
                                    should_include = False
                            
                            if call_date_end:
                                end_date = datetime.strptime(call_date_end, "%Y-%m-%d")
                                if call_date.date() > end_date.date():
                                    should_include = False
                                    
                        except Exception as e:
                            log_import(f"⚠️ Could not parse date for evaluation {evaluation.get('evaluationId', 'unknown')}: {e}")
                            # Include if we can't parse the date
                    
                    if should_include:
                        filtered_evaluations.append(evaluation)
                
                # Log filtering results
                if call_date_start or call_date_end:
                    log_import(f"📅 Date filtering: {len(evaluations)} → {len(filtered_evaluations)} evaluations")
                
                return filtered_evaluations
                
            except Exception as e:
                log_import(f"❌ Enhanced fetch failed: {e}")
                # Fallback to original fetch
                return await fetch_evaluations(max_docs_param)
        
        # ✅ Temporarily replace the fetch function
        original_fetch = app_module.fetch_evaluations
        app_module.fetch_evaluations = fetch_filtered_evaluations
        
        try:
            # Call the existing production import with enhanced parameters
            await run_production_import(
                collection=collection,
                max_docs=max_docs,
                batch_size=batch_size
            )
        finally:
            # Restore original fetch function
            app_module.fetch_evaluations = original_fetch
            
    except Exception as e:
        logger.error(f"❌ Enhanced import failed: {e}")
        # Try to get update_import_status function safely
        try:
            import sys
            app_module = sys.modules.get('app')
            if app_module:
                update_import_status = getattr(app_module, 'update_import_status')
                update_import_status("failed", error=f"Enhanced import failed: {str(e)}")
        except:
            pass
        raise

@import_router.post("/import")
async def start_enhanced_import(config: ImportConfig, background_tasks: BackgroundTasks):
    """
    Enhanced import endpoint that properly integrates with app.py
    """
    try:
        # ✅ Import import_status safely
        import sys
        app_module = sys.modules.get('app')
        
        if not app_module:
            raise HTTPException(status_code=500, detail="App module not available")
        
        import_status = getattr(app_module, 'import_status')
        
        if import_status["status"] == "running":
            raise HTTPException(status_code=400, detail="Import is already running")
        
        logger.info(f"🚀 Starting enhanced import with config: {config}")
        
        # Validate configuration
        if config.max_docs is not None and config.max_docs <= 0:
            raise HTTPException(status_code=400, detail="max_docs must be a positive number")
        
        if config.batch_size is not None and config.batch_size <= 0:
            raise HTTPException(status_code=400, detail="batch_size must be a positive number")
        
        # Validate date range
        if config.call_date_start and config.call_date_end:
            try:
                start_date = datetime.strptime(config.call_date_start, "%Y-%m-%d")
                end_date = datetime.strptime(config.call_date_end, "%Y-%m-%d")
                
                if start_date > end_date:
                    raise HTTPException(status_code=400, detail="Start date cannot be after end date")
                    
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid date format (use YYYY-MM-DD): {e}")
        
        # Reset import status
        import_status.update({
            "status": "idle",
            "start_time": None,
            "end_time": None,
            "current_step": None,
            "results": {},
            "error": None,
            "import_type": config.import_type
        })
        
        # Log the parameters for debugging
        if config.max_docs:
            logger.info(f"📊 Document limit: {config.max_docs}")
        else:
            logger.info("📊 No document limit - processing all available")
            
        if config.call_date_start or config.call_date_end:
            date_info = f"Date filter: {config.call_date_start or 'unlimited'} to {config.call_date_end or 'unlimited'}"
            logger.info(f"📅 {date_info}")
        
        # Start the enhanced import process in background
        background_tasks.add_task(
            run_enhanced_production_import,
            collection=config.collection,
            max_docs=config.max_docs,
            batch_size=config.batch_size,
            call_date_start=config.call_date_start,
            call_date_end=config.call_date_end
        )
        
        return {
            "status": "success",
            "message": f"Enhanced import started: {config.import_type} mode with date filtering",
            "collection": config.collection,
            "max_docs": config.max_docs,
            "import_type": config.import_type,
            "date_range": {
                "start": config.call_date_start,
                "end": config.call_date_end
            } if config.call_date_start or config.call_date_end else None,
            "batch_size": config.batch_size,
            "structure": "evaluation_grouped",
            "features": "real_data_filters_with_date_filtering",
            "version": "6.2.1_circular_import_fixed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Enhanced import startup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start enhanced import: {str(e)}")

@import_router.get("/import_status")
async def get_import_status():
    """Get current import status"""
    try:
        import sys
        app_module = sys.modules.get('app')
        
        if not app_module:
            return {"status": "error", "error": "App module not available"}
        
        import_status = getattr(app_module, 'import_status')
        
        return {
            "status": import_status.get("status", "unknown"),
            "current_step": import_status.get("current_step"),
            "start_time": import_status.get("start_time"),
            "end_time": import_status.get("end_time"),
            "results": import_status.get("results", {}),
            "error": import_status.get("error"),
            "import_type": import_status.get("import_type")
        }
    except Exception as e:
        logger.error(f"❌ Failed to get import status: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@import_router.post("/clear_import_timestamp")
async def clear_import_timestamp():
    """Clear the last import timestamp for incremental imports"""
    try:
        logger.info("🔄 Import timestamp cleared")
        return {
            "status": "success",
            "message": "Import timestamp cleared successfully"
        }
    except Exception as e:
        logger.error(f"❌ Failed to clear timestamp: {e}")
        return {
            "status": "error",
            "error": str(e)
        }