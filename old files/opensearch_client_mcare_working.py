# opensearch_client.py - INCREMENTAL IMPORT SUPPORT
# Enhanced with document replacement and cleanup for incremental imports
# Uses exact endpoint field names: id, url, program, name, lob, updated, content
import os
import json
from datetime import datetime
from opensearchpy import OpenSearch
from typing import List, Dict, Optional

client = OpenSearch(
    hosts=[{
        "host": os.getenv("OPENSEARCH_HOST"),
        "port": int(os.getenv("OPENSEARCH_PORT", "25060"))
    }],
    http_auth=(os.getenv("OPENSEARCH_USER"), os.getenv("OPENSEARCH_PASS")),
    use_ssl=True,
    verify_certs=False
)

async def delete_document_chunks(document_id: str, collection: str) -> int:
    """
    Delete all chunks for a specific document from OpenSearch
    This is used during incremental imports to replace updated documents
    
    Args:
        document_id: The document ID to delete chunks for
        collection: The collection name
    
    Returns:
        Number of chunks deleted
    """
    index = f"kb-{collection}"
    
    try:
        # Search for all chunks belonging to this document
        search_query = {
            "query": {
                "term": {
                    "id.keyword": document_id
                }
            },
            "size": 10000,  # Large number to get all chunks
            "_source": False  # We only need the document IDs
        }
        
        search_response = client.search(index=index, body=search_query)
        chunks_to_delete = search_response.get("hits", {}).get("hits", [])
        
        if not chunks_to_delete:
            print(f"[CLEANUP] No existing chunks found for document {document_id} in {collection}")
            return 0
        
        # Delete chunks using bulk delete
        delete_operations = []
        for chunk in chunks_to_delete:
            delete_operations.append({
                "delete": {
                    "_index": index,
                    "_id": chunk["_id"]
                }
            })
        
        if delete_operations:
            # Perform bulk delete
            bulk_body = ""
            for op in delete_operations:
                bulk_body += json.dumps(op) + "\n"
            
            delete_response = client.bulk(body=bulk_body, refresh=True)
            
            # Count successful deletions
            deleted_count = 0
            if "items" in delete_response:
                for item in delete_response["items"]:
                    if "delete" in item and item["delete"].get("status") in [200, 404]:
                        deleted_count += 1
            
            print(f"[CLEANUP] Deleted {deleted_count} existing chunks for document {document_id} in {collection}")
            return deleted_count
        
        return 0
        
    except Exception as e:
        print(f"[CLEANUP ERROR] Failed to delete chunks for document {document_id}: {e}")
        return 0

async def index_chunks(payload: dict, replace_existing: bool = True):
    """
    Index document chunks using EXACT endpoint field names
    Enhanced with document replacement support for incremental imports
    
    Expected payload fields: id, url, program, name, lob, updated, content (processed)
    
    Args:
        payload: Document payload with chunk data
        replace_existing: If True, delete existing chunks for this document first
    """
    
    # Validate required fields
    required_fields = ["id", "name", "collection", "text"]
    missing_fields = [field for field in required_fields if not payload.get(field)]
    
    if missing_fields:
        print(f"[OPENSEARCH ERROR] Missing required fields: {missing_fields}")
        print(f"[OPENSEARCH ERROR] Payload keys: {list(payload.keys())}")
        print(f"[OPENSEARCH ERROR] Payload sample: {str(payload)[:300]}...")
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Extract values using EXACT endpoint field names
    document_id = payload.get("id")  # EXACT: id (not id)
    name = payload.get("name")  # EXACT: name 
    url = payload.get("url", "")  # EXACT: url 
    program = payload.get("program", "All")
    lob = payload.get("lob", "All")  # EXACT: lob 
    collection = payload.get("collection")
    chunk_text = payload.get("text", "")
    offset = payload.get("offset", 0)
    vector = payload.get("vector", [])
    updated = payload.get("updated", "")  # EXACT: updated 
    indexed_at = payload.get("indexed_at", datetime.utcnow().isoformat())
    
    # Log the values being indexed with EXACT field names
    print(f"[OPENSEARCH] Indexing document with EXACT endpoint fields:")
    print(f"[OPENSEARCH] - id: {document_id}")
    print(f"[OPENSEARCH] - name: {name}")
    print(f"[OPENSEARCH] - url: {url}")
    print(f"[OPENSEARCH] - program: {program}")
    print(f"[OPENSEARCH] - lob: {lob}")
    print(f"[OPENSEARCH] - updated: {updated}")
    print(f"[OPENSEARCH] - collection: {collection}")
    print(f"[OPENSEARCH] - text length: {len(chunk_text)}")
    print(f"[OPENSEARCH] - vector length: {len(vector)}")
    print(f"[OPENSEARCH] - replace_existing: {replace_existing}")
    
    # NEW: Delete existing chunks if this is a replacement (incremental import)
    if replace_existing and offset == 0:  # Only do this for the first chunk to avoid multiple deletions
        deleted_count = await delete_document_chunks(document_id, collection)
        if deleted_count > 0:
            print(f"[OPENSEARCH] Replaced {deleted_count} existing chunks for document {document_id}")
    
    # Create index name and document ID
    index = f"kb-{collection}"
    doc_id = f"{document_id}_{offset}"
    
    # Build the document using EXACT endpoint field names
    doc = {
        # EXACT endpoint fields
        "id": str(document_id),
        "url": str(url),
        "program": str(program),
        "name": str(name),
        "lob": str(lob),
        "updated": str(updated),
        
        # Derived/processed fields
        "collection": str(collection),
        "chunk": {
            "text": str(chunk_text),
            "offset": int(offset)
        },
        "vector": vector,
        "indexed_at": str(indexed_at),
        "import_type": "incremental" if replace_existing else "initial"  # NEW: Track import type
    }
    
    # Log the final document structure with exact field names
    print(f"[OPENSEARCH] Final document structure (EXACT endpoint fields):")
    print(f"[OPENSEARCH] - id: {doc['id']}")
    print(f"[OPENSEARCH] - name: {doc['name']}")  
    print(f"[OPENSEARCH] - url: {doc['url']}")
    print(f"[OPENSEARCH] - program: {doc['program']}")
    print(f"[OPENSEARCH] - lob: {doc['lob']}")
    print(f"[OPENSEARCH] - updated: {doc['updated']}")
    print(f"[OPENSEARCH] - collection: {doc['collection']}")

    # Create index with EXACT endpoint field mapping if it doesn't exist
    try:
        client.indices.get(index=index)
        print(f"[OPENSEARCH] Index '{index}' already exists")
    except Exception:
        print(f"[OPENSEARCH] Creating index '{index}' with EXACT endpoint field mapping...")
        
        # Index mapping using EXACT endpoint field names
        mapping = {
            "mappings": {
                "properties": {
                    # EXACT endpoint field names
                    "id": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "url": {
                        "type": "keyword",
                        "index": True
                    },
                    "program": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        },
                        "analyzer": "keyword"  # Exact match for filtering
                    },
                    "name": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        },
                        "analyzer": "standard"
                    },
                    "lob": {
                        "type": "keyword"  # LOB IDs are exact values
                    },
                    "updated": {
                        "type": "date",
                        "format": "strict_date_optional_time||epoch_millis||yyyy-MM-dd HH:mm:ss||yyyy-MM-dd'T'HH:mm:ss||yyyy-MM-dd'T'HH:mm:ss.SSS'Z'||yyyy-MM-dd'T'HH:mm:ss.SSSSSS'Z'"
                    },
                    
                    # Derived/processed fields
                    "collection": {
                        "type": "keyword"
                    },
                    
                    # Chunk data structure (processed from content field)
                    "chunk": {
                        "properties": {
                            "text": {
                                "type": "text",
                                "analyzer": "standard",
                                "search_analyzer": "standard"
                            },
                            "offset": {"type": "integer"}
                        }
                    },
                    
                    # Vector embeddings (generated from content)
                    "vector": {
                        "type": "dense_vector",
                        "dimension": int(os.getenv("EMBEDDING_DIMENSION", "384")),
                        "index": True,
                        "similarity": "cosine"
                    },
                    
                    # Processing metadata
                    "indexed_at": {
                        "type": "date",
                        "format": "strict_date_optional_time"
                    },
                    
                    # NEW: Import tracking
                    "import_type": {
                        "type": "keyword"
                    }
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "5s"
            }
        }
        
        try:
            client.indices.create(index=index, body=mapping)
            print(f"[OPENSEARCH] Successfully created index '{index}' with EXACT endpoint field mapping")
        except Exception as e:
            print(f"[OPENSEARCH] Warning: Could not create index mapping for '{index}': {e}")
    
    # Index the document
    try:
        result = client.index(index=index, id=doc_id, body=doc)
        print(f"[OPENSEARCH] Successfully indexed document: {doc_id} in {index}")
        print(f"[OPENSEARCH] Result: {result.get('result', 'unknown')}")
        
        # Verify the document was indexed correctly with EXACT field names
        try:
            # Wait a moment for indexing to complete
            import time
            time.sleep(0.1)
            
            # Retrieve the document to verify EXACT field mapping
            retrieved = client.get(index=index, id=doc_id)
            source = retrieved.get("_source", {})
            
            print(f"[OPENSEARCH] Verification - Retrieved document with EXACT endpoint fields:")
            print(f"[OPENSEARCH] - id: {source.get('id', 'MISSING')}")
            print(f"[OPENSEARCH] - name: {source.get('name', 'MISSING')}")
            print(f"[OPENSEARCH] - url: {source.get('url', 'MISSING')}")
            print(f"[OPENSEARCH] - program: {source.get('program', 'MISSING')}")
            print(f"[OPENSEARCH] - lob: {source.get('lob', 'MISSING')}")
            print(f"[OPENSEARCH] - updated: {source.get('updated', 'MISSING')}")
            
            # Check if any EXACT endpoint fields are missing or N/A
            endpoint_fields = ["id", "name", "url", "program", "lob", "updated"]
            problematic_fields = []
            for field in endpoint_fields:
                value = source.get(field, "MISSING")
                if value in ["MISSING", "N/A", "", None]:
                    problematic_fields.append(field)
            
            if problematic_fields:
                print(f"[OPENSEARCH] WARNING: Problematic EXACT endpoint fields: {problematic_fields}")
            else:
                print(f"[OPENSEARCH] SUCCESS: All EXACT endpoint fields properly indexed")
                
        except Exception as verify_error:
            print(f"[OPENSEARCH] Could not verify document indexing: {verify_error}")
            
    except Exception as e:
        print(f"[OPENSEARCH ERROR] Failed to index document: {e}")
        print(f"[OPENSEARCH ERROR] Document ID: {document_id}")
        print(f"[OPENSEARCH ERROR] Document Name: {name}")
        print(f"[OPENSEARCH ERROR] Collection: {collection}")
        print(f"[OPENSEARCH ERROR] Payload: {json.dumps(payload, indent=2)[:500]}...")
        raise

def test_connection():
    """Test OpenSearch connection"""
    try:
        info = client.info()
        print(f"✅ OpenSearch connected: {info['version']['number']}")
        return True
    except Exception as e:
        print(f"❌ OpenSearch connection failed: {e}")
        return False

def debug_exact_field_mapping():
    """Debug function to check EXACT endpoint field mapping in indices"""
    try:
        # Get a sample document from each index
        indices_response = client.cat.indices(index="kb-*", format="json")
        print(f"[DEBUG] Found indices: {[idx['index'] for idx in indices_response]}")
        
        for idx_info in indices_response:
            index_name = idx_info['index']
            print(f"\n[DEBUG] Checking EXACT endpoint fields in index: {index_name}")
            
            # Get sample documents
            search_response = client.search(
                index=index_name,
                body={
                    "size": 1,
                    "query": {"match_all": {}}
                }
            )
            
            hits = search_response.get("hits", {}).get("hits", [])
            if hits:
                sample_doc = hits[0]["_source"]
                print(f"[DEBUG] Sample document from {index_name} - EXACT endpoint fields:")
                
                # Check EXACT endpoint fields
                endpoint_fields = ["id", "url", "program", "name", "lob", "updated"]
                for field in endpoint_fields:
                    value = sample_doc.get(field, "MISSING")
                    status = "✅" if value and value not in ["MISSING", "N/A", ""] else "❌"
                    print(f"[DEBUG]   {status} {field}: {value}")
                
                # Check derived fields
                derived_fields = ["collection"]
                for field in derived_fields:
                    value = sample_doc.get(field, "MISSING")
                    status = "✅" if value and value not in ["MISSING", "N/A", ""] else "❌"
                    print(f"[DEBUG]   {status} {field} (derived): {value}")
                
                print(f"[DEBUG]    All available fields: {list(sample_doc.keys())}")
            else:
                print(f"[DEBUG] No documents found in {index_name}")
                
    except Exception as e:
        print(f"[DEBUG] Error checking EXACT endpoint field mapping: {e}")

def get_document_count(index_pattern="kb-*"):
    """Get total document count across all knowledge base indices"""
    try:
        response = client.count(index=index_pattern)
        return response['count']
    except Exception as e:
        print(f"Error getting document count: {e}")
        return 0

def search_by_exact_id(id: str, index_pattern="kb-*"):
    """Search for a document by its EXACT endpoint ID"""
    try:
        query = {
            "query": {
                "term": {
                    "id.keyword": id  # Using EXACT endpoint field name
                }
            }
        }
        response = client.search(index=index_pattern, body=query)
        return response.get("hits", {}).get("hits", [])
    except Exception as e:
        print(f"Error searching by EXACT ID {id}: {e}")
        return []

def get_document_last_updated(document_id: str, collection: str) -> Optional[str]:
    """
    Get the last updated timestamp for a document in OpenSearch
    
    Args:
        document_id: The document ID to check
        collection: The collection name
    
    Returns:
        Last updated timestamp string or None if not found
    """
    index = f"kb-{collection}"
    
    try:
        search_query = {
            "query": {
                "term": {
                    "id.keyword": document_id
                }
            },
            "size": 1,
            "_source": ["updated"],
            "sort": [
                {
                    "indexed_at": {
                        "order": "desc"
                    }
                }
            ]
        }
        
        response = client.search(index=index, body=search_query)
        hits = response.get("hits", {}).get("hits", [])
        
        if hits:
            return hits[0]["_source"].get("updated")
        
        return None
        
    except Exception as e:
        print(f"[OPENSEARCH] Error getting last updated for document {document_id}: {e}")
        return None

def cleanup_orphaned_chunks(max_age_days: int = 30) -> int:
    """
    Clean up orphaned chunks that are older than specified days
    This helps maintain index health during incremental imports
    
    Args:
        max_age_days: Maximum age in days for chunks to be considered for cleanup
    
    Returns:
        Number of chunks cleaned up
    """
    try:
        from datetime import datetime, timedelta
        cutoff_date = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat()
        
        # Search for old chunks
        cleanup_query = {
            "query": {
                "range": {
                    "indexed_at": {
                        "lt": cutoff_date
                    }
                }
            },
            "size": 1000,  # Process in batches
            "_source": False
        }
        
        total_cleaned = 0
        
        # Process all indices
        indices_response = client.cat.indices(index="kb-*", format="json")
        
        for idx_info in indices_response:
            index_name = idx_info['index']
            
            try:
                search_response = client.search(index=index_name, body=cleanup_query)
                old_chunks = search_response.get("hits", {}).get("hits", [])
                
                if old_chunks:
                    # Delete old chunks
                    delete_operations = []
                    for chunk in old_chunks:
                        delete_operations.append({
                            "delete": {
                                "_index": index_name,
                                "_id": chunk["_id"]
                            }
                        })
                    
                    if delete_operations:
                        bulk_body = ""
                        for op in delete_operations:
                            bulk_body += json.dumps(op) + "\n"
                        
                        delete_response = client.bulk(body=bulk_body, refresh=True)
                        
                        # Count successful deletions
                        deleted_count = 0
                        if "items" in delete_response:
                            for item in delete_response["items"]:
                                if "delete" in item and item["delete"].get("status") in [200, 404]:
                                    deleted_count += 1
                        
                        total_cleaned += deleted_count
                        print(f"[CLEANUP] Cleaned up {deleted_count} old chunks from {index_name}")
                
            except Exception as e:
                print(f"[CLEANUP] Error cleaning up {index_name}: {e}")
                continue
        
        if total_cleaned > 0:
            print(f"[CLEANUP] Total cleanup: {total_cleaned} old chunks removed")
        
        return total_cleaned
        
    except Exception as e:
        print(f"[CLEANUP ERROR] Failed to cleanup orphaned chunks: {e}")
        return 0

def get_import_statistics() -> Dict:
    """
    Get statistics about the current state of the indices for import monitoring
    
    Returns:
        Dictionary with import statistics
    """
    try:
        stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "collections": {},
            "import_types": {},
            "oldest_document": None,
            "newest_document": None
        }
        
        # Aggregation query to get comprehensive stats
        agg_query = {
            "size": 0,
            "aggs": {
                "collections": {
                    "terms": {"field": "_index", "size": 50},
                    "aggs": {
                        "unique_documents": {
                            "cardinality": {"field": "id.keyword"}
                        },
                        "import_types": {
                            "terms": {"field": "import_type", "size": 10, "missing": "unknown"}
                        }
                    }
                },
                "import_types_overall": {
                    "terms": {"field": "import_type", "size": 10, "missing": "unknown"}
                },
                "date_range": {
                    "stats": {"field": "indexed_at"}
                }
            }
        }
        
        response = client.search(index="kb-*", body=agg_query)
        aggs = response.get("aggregations", {})
        
        # Overall stats
        stats["total_chunks"] = response.get("hits", {}).get("total", {}).get("value", 0)
        
        # Collection stats
        for coll_bucket in aggs.get("collections", {}).get("buckets", []):
            collection = coll_bucket["key"].replace("kb-", "")
            stats["collections"][collection] = {
                "chunks": coll_bucket["doc_count"],
                "unique_documents": coll_bucket["unique_documents"]["value"],
                "import_types": {}
            }
            
            # Import types per collection
            for import_bucket in coll_bucket.get("import_types", {}).get("buckets", []):
                import_type = import_bucket["key"]
                stats["collections"][collection]["import_types"][import_type] = import_bucket["doc_count"]
        
        # Overall import types
        for import_bucket in aggs.get("import_types_overall", {}).get("buckets", []):
            stats["import_types"][import_bucket["key"]] = import_bucket["doc_count"]
        
        # Date range
        date_stats = aggs.get("date_range", {})
        if date_stats.get("min"):
            stats["oldest_document"] = datetime.fromtimestamp(date_stats["min"] / 1000).isoformat()
        if date_stats.get("max"):
            stats["newest_document"] = datetime.fromtimestamp(date_stats["max"] / 1000).isoformat()
        
        # Calculate total unique documents across all collections
        stats["total_documents"] = sum(
            coll_data["unique_documents"] for coll_data in stats["collections"].values()
        )
        
        return stats
        
    except Exception as e:
        print(f"[STATS ERROR] Failed to get import statistics: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Test the OpenSearch client with EXACT endpoint field mapping and incremental features
    print("Testing OpenSearch client with EXACT endpoint field mapping and incremental import support...")
    print("Expected fields: id, url, program, name, lob, updated, content")
    
    if test_connection():
        count = get_document_count()
        print(f"Total documents in knowledge base: {count}")
        
        # Debug EXACT endpoint field mapping
        debug_exact_field_mapping()
        
        # Test incremental import features
        print("\n🔄 Testing incremental import features...")
        
        # Get import statistics
        stats = get_import_statistics()
        if "error" not in stats:
            print(f"📊 Import Statistics:")
            print(f"   Total documents: {stats['total_documents']}")
            print(f"   Total chunks: {stats['total_chunks']}")
            print(f"   Collections: {len(stats['collections'])}")
            print(f"   Import types: {stats['import_types']}")
            if stats.get('oldest_document'):
                print(f"   Oldest document: {stats['oldest_document']}")
            if stats.get('newest_document'):
                print(f"   Newest document: {stats['newest_document']}")
        
        print("✅ OpenSearch client test completed with EXACT endpoint field names and incremental support!")
    else:
        print("❌ Fix OpenSearch configuration before running the main application!")