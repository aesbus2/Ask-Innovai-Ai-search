# opensearch_debug.py - FIXED VERSION
# This fixes the timeout configuration error

import os
import sys
import logging
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from opensearchpy import OpenSearch
    from opensearchpy.exceptions import ConnectionError, RequestError
    OPENSEARCH_AVAILABLE = True
except ImportError:
    print("❌ opensearch-py not installed. Run: pip install opensearch-py")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_fixed_client():
    """Create OpenSearch client with FIXED timeout configuration"""
    try:
        # FIXED: Use integers for timeouts, not strings
        client = OpenSearch(
            hosts=[{
                "host": os.getenv("OPENSEARCH_HOST"),
                "port": int(os.getenv("OPENSEARCH_PORT", "25060"))
            }],
            http_auth=(os.getenv("OPENSEARCH_USER"), os.getenv("OPENSEARCH_PASS")),
            use_ssl=True,
            verify_certs=False,
            request_timeout=30,        # INTEGER - not "30s"
            connect_timeout=10,        # INTEGER - not "10s"  
            max_retries=3,
            retry_on_timeout=True,
            ssl_show_warn=False
        )
        
        return client
        
    except Exception as e:
        logger.error(f"Failed to create client: {e}")
        return None

def test_basic_connection():
    """Test 1: Basic connection"""
    print("1. Basic Connection Test:")
    print("   Creating client...")
    
    client = create_fixed_client()
    if not client:
        print("   ❌ Failed to create client")
        return False
    
    try:
        # FIXED: Use integer timeout
        result = client.ping(request_timeout=5)  # 5 seconds, not "5s"
        if result:
            print("   ✅ Basic connection successful!")
            return True
        else:
            print("   ❌ Ping returned False")
            return False
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False

def test_cluster_info():
    """Test 2: Cluster info"""
    print("2. Cluster Info Test:")
    
    client = create_fixed_client()
    if not client:
        print("   ❌ No client available")
        return
    
    try:
        # FIXED: Use integer timeout
        info = client.info(request_timeout=10)
        print(f"   ✅ Cluster: {info.get('cluster_name', 'Unknown')}")
        print(f"   ✅ Version: {info.get('version', {}).get('number', 'Unknown')}")
        
        # Test health
        health = client.cluster.health(request_timeout=10)
        print(f"   ✅ Health: {health.get('status', 'Unknown')}")
        
    except Exception as e:
        print(f"   ❌ Cluster info failed: {e}")

def test_indices():
    """Test 3: Check for evaluation indices"""
    print("3. Indices Check:")
    
    client = create_fixed_client()
    if not client:
        print("   ❌ No client available")
        return
    
    try:
        # FIXED: Use integer timeout
        indices = client.cat.indices(index="eval-*", format="json", request_timeout=10)
        if indices:
            print(f"   ✅ Found {len(indices)} evaluation indices")
            for idx in indices[:3]:  # Show first 3
                print(f"      - {idx.get('index')}: {idx.get('docs.count', 0)} docs")
        else:
            print("   ⚠️ No evaluation indices found")
            
    except Exception as e:
        print(f"   ❌ Indices check failed: {e}")

def test_document_indexing():
    """Test 4: Document indexing"""
    print("4. Document Indexing Test:")
    
    client = create_fixed_client()
    if not client:
        print("   ❌ No client available")
        return
    
    try:
        # Create test document
        test_doc = {
            "test_field": "test_value",
            "timestamp": datetime.now().isoformat(),
            "debug_test": True
        }
        
        print(f"   Testing doc ID: test-{int(datetime.now().timestamp())}")
        
        # FIXED: Use integer timeout
        result = client.index(
            index="connection-test",
            id=f"test-{int(datetime.now().timestamp())}",
            body=test_doc,
            request_timeout=15  # INTEGER timeout
        )
        
        print(f"   ✅ Document indexed successfully!")
        print(f"   Result: {result.get('result', 'Unknown')}")
        
        # Clean up
        client.delete(
            index="connection-test", 
            id=f"test-{int(datetime.now().timestamp())}", 
            request_timeout=5,
            ignore=[404]  # Ignore if already deleted
        )
        
    except Exception as e:
        print(f"   ❌ Document indexing test failed: {e}")

def test_search():
    """Test 5: Basic search"""
    print("5. Search Test:")
    
    client = create_fixed_client()
    if not client:
        print("   ❌ No client available")
        return
    
    try:
        # FIXED: Use integer timeout and proper query structure
        search_body = {
            "query": {"match_all": {}},
            "size": 3,
            "timeout": "10s"  # Query timeout as string (this is correct)
        }
        
        result = client.search(
            index="eval-*",
            body=search_body,
            request_timeout=15  # Client timeout as integer
        )
        
        hits = result.get("hits", {}).get("hits", [])
        print(f"   ✅ Search successful! Found {len(hits)} results")
        
        if hits:
            for i, hit in enumerate(hits):
                source = hit.get("_source", {})
                print(f"      {i+1}. ID: {source.get('evaluationId', 'Unknown')}")
                print(f"         Template: {source.get('template_name', 'Unknown')}")
        
    except Exception as e:
        print(f"   ❌ Search test failed: {e}")

def test_aggregation():
    """Test 6: Safe aggregation"""
    print("6. Safe Aggregation Test:")
    
    client = create_fixed_client()
    if not client:
        print("   ❌ No client available")
        return
    
    try:
        # FIXED: Use safe aggregation query
        agg_body = {
            "size": 0,
            "timeout": "15s",
            "aggs": {
                "index_count": {
                    "terms": {
                        "field": "_index",
                        "size": 10
                    }
                }
            }
        }
        
        result = client.search(
            index="eval-*",
            body=agg_body,
            request_timeout=20  # INTEGER timeout
        )
        
        if "aggregations" in result:
            buckets = result["aggregations"]["index_count"]["buckets"]
            print(f"   ✅ Aggregation successful! Found {len(buckets)} index groups")
            for bucket in buckets:
                print(f"      - {bucket['key']}: {bucket['doc_count']} docs")
        else:
            print("   ⚠️ No aggregation results")
            
    except Exception as e:
        print(f"   ❌ Aggregation test failed: {e}")

def main():
    """Run all tests"""
    print("🧪 OpenSearch Debug Script - FIXED VERSION")
    print("=" * 50)
    
    # Check environment
    host = os.getenv("OPENSEARCH_HOST")
    port = os.getenv("OPENSEARCH_PORT", "25060")
    user = os.getenv("OPENSEARCH_USER")
    
    if not host or not user:
        print("❌ Missing environment variables:")
        print(f"   OPENSEARCH_HOST: {host or 'NOT SET'}")
        print(f"   OPENSEARCH_PORT: {port}")
        print(f"   OPENSEARCH_USER: {user or 'NOT SET'}")
        print("   OPENSEARCH_PASS: {'SET' if os.getenv('OPENSEARCH_PASS') else 'NOT SET'}")
        return
    
    print(f"🔧 Configuration:")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   User: {user}")
    print(f"   SSL: True")
    print()
    
    # Run tests
    connection_ok = test_basic_connection()
    print()
    
    if connection_ok:
        test_cluster_info()
        print()
        test_indices()
        print()
        test_document_indexing()
        print()
        test_search()
        print()
        test_aggregation()
        print()
        
        print("✅ All tests completed!")
        print("🔧 If you saw any errors, they should now be fixed.")
        
    else:
        print("❌ Basic connection failed - check your configuration")
        
    print("=" * 50)

if __name__ == "__main__":
    main()