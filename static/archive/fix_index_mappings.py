#!/usr/bin/env python3
"""
fix_index_mappings.py - Script to verify and fix OpenSearch index mappings
Ensures field mappings match the endpoint structure: id, url, program, name, lob, updated, content
"""

import os
import json
from opensearchpy import OpenSearch
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenSearch client
client = OpenSearch(
    hosts=[{
        "host": os.getenv("OPENSEARCH_HOST"),
        "port": int(os.getenv("OPENSEARCH_PORT", "25060"))
    }],
    http_auth=(os.getenv("OPENSEARCH_USER"), os.getenv("OPENSEARCH_PASS")),
    use_ssl=True,
    verify_certs=False
)

def get_correct_mapping():
    """Define the correct index mapping for EXACT endpoint data fields"""
    return {
        "mappings": {
            "properties": {
                # EXACT endpoint field names: id, url, program, name, lob, updated, content
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
                "lob_name": {
                    "type": "text",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    },
                    "analyzer": "keyword"  # Exact match for filtering
                },
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
                }
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "5s"
        }
    }

def check_existing_indices():
    """Check current indices and their mappings"""
    print("🔍 CHECKING EXISTING INDICES")
    print("=" * 50)
    
    try:
        indices = client.indices.get(index="kb-*")
        print(f"📊 Found {len(indices)} indices:")
        
        issues_found = []
        
        for index_name, index_info in indices.items():
            print(f"\n📁 Index: {index_name}")
            
            # Get current mapping
            current_mapping = index_info.get("mappings", {}).get("properties", {})
            
            # Check required fields using EXACT endpoint field names
            required_fields = {
                "id": "text",           # EXACT: id
                "url": "keyword",       # EXACT: url
                "program": "text",      # EXACT: program
                "name": "text",         # EXACT: name
                "lob": "keyword",       # EXACT: lob
                "updated": "date",      # EXACT: updated
                "lob_name": "text",     # Derived from lob
                "collection": "keyword",
                "chunk": "object",
                "vector": "dense_vector",
                "indexed_at": "date"
            }
            
            print(f"   🔧 EXACT Endpoint Field Mapping Check:")
            for field, expected_type in required_fields.items():
                if field in current_mapping:
                    actual_type = current_mapping[field].get("type", "unknown")
                    if expected_type == "object" and "properties" in current_mapping[field]:
                        actual_type = "object"
                    
                    if actual_type == expected_type or (expected_type == "text" and actual_type in ["text", "keyword"]):
                        print(f"      ✅ {field}: {actual_type}")
                    else:
                        print(f"      ❌ {field}: {actual_type} (expected: {expected_type})")
                        issues_found.append(f"{index_name}.{field}")
                else:
                    print(f"      ❌ {field}: MISSING")
                    issues_found.append(f"{index_name}.{field}")
            
            # Check for vector dimension
            if "vector" in current_mapping:
                vector_dim = current_mapping["vector"].get("dimension", "unknown")
                expected_dim = int(os.getenv("EMBEDDING_DIMENSION", "384"))
                if vector_dim == expected_dim:
                    print(f"      ✅ vector dimension: {vector_dim}")
                else:
                    print(f"      ❌ vector dimension: {vector_dim} (expected: {expected_dim})")
                    issues_found.append(f"{index_name}.vector.dimension")
            
            # Get sample document to check actual data
            try:
                sample_response = client.search(
                    index=index_name,
                    body={"size": 1, "query": {"match_all": {}}},
                    _source=True
                )
                
                hits = sample_response.get("hits", {}).get("hits", [])
                if hits:
                    sample_doc = hits[0]["_source"]
                    print(f"   📄 Sample Document Fields (EXACT endpoint mapping):")
                    
                    for field in required_fields.keys():
                        if field == "chunk":
                            chunk_data = sample_doc.get("chunk", {})
                            if isinstance(chunk_data, dict) and "text" in chunk_data:
                                print(f"      ✅ {field}: {type(chunk_data).__name__} with text")
                            else:
                                print(f"      ❌ {field}: {type(chunk_data).__name__} - invalid structure")
                                issues_found.append(f"{index_name}.{field}.structure")
                        else:
                            value = sample_doc.get(field)
                            if value and value != "N/A":
                                print(f"      ✅ {field}: {type(value).__name__} = {str(value)[:30]}...")
                            else:
                                print(f"      ❌ {field}: {value or 'MISSING'}")
                                issues_found.append(f"{index_name}.{field}.data")
                else:
                    print(f"   📄 No documents found in {index_name}")
                    
            except Exception as e:
                print(f"   ❌ Error checking sample document: {e}")
        
        return indices, issues_found
        
    except Exception as e:
        print(f"❌ Error checking indices: {e}")
        return {}, []

def fix_index_mapping_issues(indices, issues_found):
    """Fix index mapping issues"""
    print(f"\n🔧 FIXING INDEX MAPPING ISSUES")
    print("=" * 50)
    
    if not issues_found:
        print("✅ No issues found! All mappings are correct.")
        return
    
    print(f"🚨 Found {len(issues_found)} issues:")
    for issue in issues_found:
        print(f"   - {issue}")
    
    # Group issues by index
    index_issues = {}
    for issue in issues_found:
        index_name = issue.split('.')[0]
        if index_name not in index_issues:
            index_issues[index_name] = []
        index_issues[index_name].append(issue)
    
    print(f"\n🔄 Recommended Actions:")
    
    for index_name, index_issue_list in index_issues.items():
        print(f"\n📁 {index_name}:")
        
        # Check if issues are mapping-related or data-related
        mapping_issues = [i for i in index_issue_list if not i.endswith('.data')]
        data_issues = [i for i in index_issue_list if i.endswith('.data')]
        
        if mapping_issues:
            print(f"   🔧 Mapping Issues: {len(mapping_issues)}")
            print(f"   💡 Recommendation: Recreate index with correct mapping")
            print(f"   ⚠️  This will delete existing data in {index_name}")
            
            # Ask user if they want to recreate the index
            response = input(f"   ❓ Recreate {index_name} with correct mapping? (y/N): ").lower()
            if response == 'y':
                recreate_index(index_name)
            else:
                print(f"   ⏭️  Skipping {index_name}")
        
        if data_issues:
            print(f"   📄 Data Issues: {len(data_issues)}")
            print(f"   💡 Recommendation: Re-import data to populate missing fields")
            print(f"   🔄 Run import process with updated field mapping logic")

def recreate_index(index_name):
    """Recreate an index with correct mapping"""
    print(f"\n🔄 Recreating {index_name}...")
    
    try:
        # Check if index exists and get document count
        doc_count = 0
        try:
            count_response = client.count(index=index_name)
            doc_count = count_response.get("count", 0)
            print(f"   📊 Current documents: {doc_count}")
        except:
            pass
        
        # Delete existing index
        if doc_count > 0:
            print(f"   ⚠️  Deleting {doc_count} documents...")
        
        client.indices.delete(index=index_name)
        print(f"   🗑️  Deleted index: {index_name}")
        
        # Create new index with correct mapping
        correct_mapping = get_correct_mapping()
        client.indices.create(index=index_name, body=correct_mapping)
        print(f"   ✅ Created index with correct mapping: {index_name}")
        
        # Verify the new mapping
        new_mapping = client.indices.get_mapping(index=index_name)
        properties = new_mapping[index_name]["mappings"]["properties"]
        print(f"   🔍 Verified fields: {list(properties.keys())}")
        
        if doc_count > 0:
            print(f"   💡 Next: Re-import {doc_count} documents using the corrected import process")
        
    except Exception as e:
        print(f"   ❌ Error recreating index: {e}")

def create_index_template():
    """Create an index template for future kb-* indices"""
    print(f"\n📋 CREATING INDEX TEMPLATE")
    print("=" * 50)
    
    template_name = "kb-template"
    
    try:
        template_body = {
            "index_patterns": ["kb-*"],
            "priority": 1,
            "template": get_correct_mapping(),
            "version": 1,
            "_meta": {
                "description": "Template for knowledge base indices with EXACT endpoint field mapping",
                "created_by": "fix_index_mappings.py",
                "schema_version": "2.0",
                "endpoint_fields": "id, url, program, name, lob, updated, content"
            }
        }
        
        # Create or update the template
        client.indices.put_index_template(name=template_name, body=template_body)
        print(f"✅ Created/updated index template: {template_name}")
        print(f"   📋 Pattern: kb-*")
        print(f"   🔧 All new kb-* indices will use correct field mapping")
        
    except Exception as e:
        print(f"❌ Error creating index template: {e}")

def validate_field_consistency():
    """Validate that search queries will work with current field structure"""
    print(f"\n🔍 VALIDATING FIELD CONSISTENCY FOR SEARCH")
    print("=" * 50)
    
    test_queries = [
        # Test program filtering using EXACT field names
        {
            "name": "Program Filter Test (EXACT: program.keyword)",
            "query": {
                "query": {
                    "terms": {"program.keyword": ["Metro", "ASW", "All"]}
                },
                "size": 1
            }
        },
        # Test document name search using EXACT field names
        {
            "name": "Document Name Search Test (EXACT: name)", 
            "query": {
                "query": {
                    "match": {"name": "test"}
                },
                "size": 1
            }
        },
        # Test document ID search using EXACT field names
        {
            "name": "Document ID Search Test (EXACT: id.keyword)",
            "query": {
                "query": {
                    "exists": {"field": "id"}
                },
                "size": 1
            }
        },
        # Test URL field using EXACT field names
        {
            "name": "URL Field Test (EXACT: url)",
            "query": {
                "query": {
                    "exists": {"field": "url"}
                },
                "size": 1
            }
        },
        # Test vector similarity (mock)
        {
            "name": "Vector Field Test",
            "query": {
                "query": {"exists": {"field": "vector"}},
                "size": 1
            }
        }
    ]
    
    for test in test_queries:
        print(f"\n🧪 {test['name']}:")
        try:
            response = client.search(index="kb-*", body=test["query"])
            hit_count = response.get("hits", {}).get("total", {}).get("value", 0)
            print(f"   ✅ Query successful: {hit_count} results")
            
            # Check EXACT endpoint field values in results
            hits = response.get("hits", {}).get("hits", [])
            if hits:
                sample_fields = hits[0]["_source"]
                # Check EXACT endpoint fields
                for field in ["id", "name", "url", "program", "lob", "updated"]:
                    value = sample_fields.get(field, "MISSING")
                    status = "✅" if value and value != "N/A" else "❌"
                    print(f"      {status} {field} (EXACT): {value}")
                
                # Check derived fields
                for field in ["lob_name", "collection"]:
                    value = sample_fields.get(field, "MISSING")
                    status = "✅" if value and value != "N/A" else "❌"
                    print(f"      {status} {field} (derived): {value}")
            
        except Exception as e:
            print(f"   ❌ Query failed: {e}")

def main():
    """Main function to check and fix index mappings"""
    print("🔧 INDEX MAPPING FIXER - EXACT ENDPOINT FIELDS")
    print("=" * 60)
    print("This script will check and fix OpenSearch index mappings")
    print("to match the EXACT endpoint structure: id, url, program, name, lob, updated, content")
    print("=" * 60)
    
    # Test connection
    try:
        info = client.info()
        print(f"✅ Connected to OpenSearch: {info['version']['number']}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Check existing indices
    indices, issues_found = check_existing_indices()
    
    # Fix issues if found
    if indices:
        fix_index_mapping_issues(indices, issues_found)
        
        # Create template for future indices
        create_index_template()
        
        # Validate field consistency
        validate_field_consistency()
    
    print(f"\n🎯 SUMMARY")
    print("=" * 50)
    print("✅ EXACT endpoint field mapping check completed")
    print("💡 Next steps:")
    print("   1. Run a test import with 1-2 documents")
    print("   2. Check that EXACT endpoint fields are populated correctly:")
    print("      - id, url, program, name, lob, updated should NOT be 'N/A'")
    print("   3. Test search functionality with exact field names")
    print("   4. Run full import if everything looks good")

if __name__ == "__main__":
    main()