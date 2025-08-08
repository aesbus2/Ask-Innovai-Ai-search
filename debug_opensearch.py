#!/usr/bin/env python3
"""
OpenSearch Connection Debugger & Fixer
Helps diagnose and resolve OpenSearch connection issues
"""

import os
import requests
import time
import json
from opensearchpy import OpenSearch

def debug_opensearch_connection():
    """Comprehensive OpenSearch connection debugging"""
    print("🔍 OpenSearch Connection Diagnostics")
    print("=" * 50)
    
    # 1. Environment Variables Check
    print("\n1. Environment Variables:")
    opensearch_host = os.getenv("OPENSEARCH_HOST", "NOT_SET")
    opensearch_port = os.getenv("OPENSEARCH_PORT", "NOT_SET")
    opensearch_user = os.getenv("OPENSEARCH_USER", "admin")
    opensearch_pass = os.getenv("OPENSEARCH_PASS", "NOT_SET")
    
    print(f"   OPENSEARCH_HOST: {opensearch_host}")
    print(f"   OPENSEARCH_PORT: {opensearch_port}")
    print(f"   OPENSEARCH_USER: {opensearch_user}")
    print(f"   OPENSEARCH_PASS: {'SET' if opensearch_pass != 'NOT_SET' else 'NOT_SET'}")
    
    # 2. URL Construction
    print("\n2. URL Construction:")
    if opensearch_host == "NOT_SET":
        print("   ❌ OPENSEARCH_HOST not set!")
        return False
    
    # Handle port
    if opensearch_port == "NOT_SET":
        port = 25060  # Default for Digital Ocean
        print(f"   ⚠️  OPENSEARCH_PORT not set, using default: {port}")
    else:
        port = int(opensearch_port)
        print(f"   ✅ OPENSEARCH_PORT: {port}")
    
    # Remove protocol if included in host
    clean_host = opensearch_host
    if clean_host.startswith("http://"):
        clean_host = clean_host.replace("http://", "")
        protocol = "http"
    elif clean_host.startswith("https://"):
        clean_host = clean_host.replace("https://", "")
        protocol = "https"
    else:
        # Auto-detect protocol
        protocol = "https" if "cloud" in clean_host.lower() or "digitalocean" in clean_host.lower() else "http"
    
    opensearch_url = f"{protocol}://{clean_host}:{port}"
    print(f"   Final URL: {opensearch_url}")
    
    # 3. Basic Network Connectivity
    print("\n3. Network Connectivity Test:")
    try:
        # Test basic HTTP connectivity
        test_url = f"{protocol}://{clean_host}:{port}"
        print(f"   Testing: {test_url}")
        
        response = requests.get(
            test_url,
            auth=(opensearch_user, opensearch_pass),
            verify=False,
            timeout=10
        )
        
        print(f"   ✅ HTTP Response: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Basic connectivity successful")
        else:
            print(f"   ⚠️  HTTP Status: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            
    except requests.exceptions.ConnectTimeout:
        print("   ❌ Connection timeout - host may be unreachable")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"   ❌ Connection error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Network test failed: {e}")
        return False
    
    # 4. OpenSearch Client Test
    print("\n4. OpenSearch Client Test:")
    try:
        client = OpenSearch(
            hosts=[opensearch_url],
            http_auth=(opensearch_user, opensearch_pass),
            use_ssl=opensearch_url.startswith("https"),
            verify_certs=False,
            timeout=15,
            max_retries=2,
            retry_on_timeout=True
        )
        
        print("   ✅ OpenSearch client created")
        
        # Test ping
        ping_result = client.ping()
        print(f"   Ping result: {ping_result}")
        
        if ping_result:
            print("   ✅ OpenSearch ping successful")
        else:
            print("   ❌ OpenSearch ping failed")
            return False
        
        # Test cluster info
        try:
            cluster_info = client.info()
            print(f"   ✅ Cluster info: {cluster_info.get('version', {}).get('number', 'Unknown')}")
        except Exception as e:
            print(f"   ⚠️  Could not get cluster info: {e}")
        
        # Test index operations
        try:
            indices = client.indices.get_alias(index="*")
            user_indices = [name for name in indices.keys() if not name.startswith('.')]
            print(f"   ✅ Found {len(user_indices)} user indices")
            if user_indices:
                print(f"   Indices: {user_indices[:5]}...")  # Show first 5
        except Exception as e:
            print(f"   ⚠️  Could not list indices: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ OpenSearch client test failed: {e}")
        return False

def test_document_indexing():
    """Test actual document indexing"""
    print("\n5. Document Indexing Test:")
    
    # Get environment variables
    opensearch_host = os.getenv("OPENSEARCH_HOST", "NOT_SET")
    opensearch_port = int(os.getenv("OPENSEARCH_PORT", "25060"))
    opensearch_user = os.getenv("OPENSEARCH_USER", "admin")
    opensearch_pass = os.getenv("OPENSEARCH_PASS", "admin")
    
    if opensearch_host == "NOT_SET":
        print("   ❌ OPENSEARCH_HOST not set, skipping indexing test")
        return False
    
    # Build URL
    clean_host = opensearch_host.replace("http://", "").replace("https://", "")
    protocol = "https" if "cloud" in clean_host.lower() or "digitalocean" in clean_host.lower() else "http"
    opensearch_url = f"{protocol}://{clean_host}:{opensearch_port}"
    
    try:
        client = OpenSearch(
            hosts=[opensearch_url],
            http_auth=(opensearch_user, opensearch_pass),
            use_ssl=opensearch_url.startswith("https"),
            verify_certs=False,
            timeout=15,
            max_retries=2,
            retry_on_timeout=True
        )
        
        # Test document
        test_doc = {
            "text": "This is a test document for connection debugging",
            "metadata": {
                "test": True,
                "created_at": time.time()
            },
            "indexed_at": time.time()
        }
        
        test_index = "connection-test"
        test_doc_id = f"test-{int(time.time())}"
        
        print(f"   Testing index: {test_index}")
        print(f"   Testing doc ID: {test_doc_id}")
        
        # Index the document
        result = client.index(
            index=test_index,
            id=test_doc_id,
            body=test_doc,
            timeout='10'
        )
        
        print(f"   ✅ Document indexed successfully")
        print(f"   Result: {result.get('result', 'unknown')}")
        
        # Try to retrieve it
        time.sleep(1)  # Wait for indexing
        try:
            retrieved = client.get(index=test_index, id=test_doc_id)
            print(f"   ✅ Document retrieved successfully")
            
            # Clean up
            client.delete(index=test_index, id=test_doc_id)
            print(f"   ✅ Test document cleaned up")
            
        except Exception as e:
            print(f"   ⚠️  Could not retrieve test document: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Document indexing test failed: {e}")
        return False

def audit_missing_content(client):
    """Audit OpenSearch documents missing usable content"""
    print("\n8. Content Completeness Audit:")
    print("=" * 50)

    try:
        query = {
            "query": {
                "bool": {
                    "must_not": [
                        {"exists": {"field": "transcript_text"}},
                        {"exists": {"field": "chunks.embedding"}}
                    ]
                }
            },
            "_source": [
                "evaluationId",
                "template_name",
                "metadata.agentName",
                "metadata.disposition",
                "full_text"
            ]
        }

        response = client.search(index="eval-template-*", body=query, size=25)
        hits = response["hits"]["hits"]
        total = response["hits"]["total"]["value"]

        print(f"   ❗ Found {total} documents missing both `transcript_text` and `chunks.embedding`")
        if hits:
            for hit in hits:
                source = hit["_source"]
                eval_id = source.get("evaluationId", "N/A")
                agent = source.get("metadata", {}).get("agentName", "N/A")
                disposition = source.get("metadata", {}).get("disposition", "N/A")
                print(f"   - Eval ID: {eval_id}, Agent: {agent}, Disposition: {disposition}")

        return total

    except Exception as e:
        print(f"   ❌ Failed to audit content completeness: {e}")
        return 0


def provide_recommendations():
    """Provide troubleshooting recommendations"""
    print("\n6. Troubleshooting Recommendations:")
    print("=" * 50)
    
    opensearch_host = os.getenv("OPENSEARCH_HOST", "NOT_SET")
    opensearch_port = os.getenv("OPENSEARCH_PORT", "NOT_SET")
    opensearch_pass = os.getenv("OPENSEARCH_PASS", "NOT_SET")
    
    recommendations = []
    
    # Check environment variables
    if opensearch_host == "NOT_SET":
        recommendations.append("Set OPENSEARCH_HOST environment variable")
    
    if opensearch_port == "NOT_SET":
        recommendations.append("Set OPENSEARCH_PORT environment variable (usually 25060 for Digital Ocean)")
    
    if opensearch_pass == "NOT_SET":
        recommendations.append("Set OPENSEARCH_PASS environment variable")
    
    # Common issues
    recommendations.extend([
        "Verify OpenSearch cluster is running and accessible",
        "Check firewall rules and network connectivity",
        "Confirm OpenSearch credentials are correct",
        "Try increasing timeout values in opensearch_client.py",
        "Check OpenSearch cluster health and available resources",
        "Verify SSL/TLS configuration matches your cluster setup"
    ])
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    print("\n7. Next Steps:")
    print("   - Fix any environment variable issues")
    print("   - Test connection using this script")
    print("   - Check OpenSearch cluster logs for errors")
    print("   - Consider reducing batch sizes in import process")

def main():
    """Main debugging function"""
    print("🔧 OpenSearch Connection Debugger")
    print("This tool will help diagnose OpenSearch connection issues")
    print()
    
    # Run diagnostics
    connection_ok = debug_opensearch_connection()
    
    if connection_ok:
        print("\n✅ Basic connection successful!")

        # Create OpenSearch client again to reuse
        opensearch_host = os.getenv("OPENSEARCH_HOST", "NOT_SET")
        opensearch_port = int(os.getenv("OPENSEARCH_PORT", "25060"))
        opensearch_user = os.getenv("OPENSEARCH_USER", "admin")
        opensearch_pass = os.getenv("OPENSEARCH_PASS", "admin")
        clean_host = opensearch_host.replace("http://", "").replace("https://", "")
        protocol = "https" if "cloud" in clean_host.lower() or "digitalocean" in clean_host.lower() else "http"
        opensearch_url = f"{protocol}://{clean_host}:{opensearch_port}"

        client = OpenSearch(
            hosts=[opensearch_url],
            http_auth=(opensearch_user, opensearch_pass),
            use_ssl=opensearch_url.startswith("https"),
            verify_certs=False,
            timeout=15,
            max_retries=2,
            retry_on_timeout=True
        )

        audit_missing_content(client)
        indexing_ok = test_document_indexing()
        
        if indexing_ok:
            print("\n🎉 All tests passed! OpenSearch connection is working.")
            print("The issue may be with:")
            print("   - High load during import")
            print("   - Timeout settings too aggressive")
            print("   - Network instability")
            print("   - OpenSearch cluster resource limits")
        else:
            print("\n⚠️  Connection works but indexing failed")
            provide_recommendations()
    else:
        print("\n❌ Connection failed")
        provide_recommendations()

if __name__ == "__main__":
    main()