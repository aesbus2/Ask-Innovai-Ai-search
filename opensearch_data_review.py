#!/usr/bin/env python3
"""
OpenSearch Structure Inspector & Cleanup Tool
Simple script to check OpenSearch indices, mappings, and document counts
Includes safe cleanup functionality for your Ask InnovAI project
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any

# Check for required dependencies
try:
    from opensearchpy import OpenSearch
    from opensearchpy.exceptions import ConnectionError, RequestError
except ImportError:
    print("❌ opensearch-py not installed!")
    print("📦 Install with: pip install opensearch-py")
    sys.exit(1)

# Load environment variables (you can also set these directly in the script)
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "25060"))
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "admin")
OPENSEARCH_PASS = os.getenv("OPENSEARCH_PASS", "admin")

def create_opensearch_client():
    """Create OpenSearch client using your exact configuration"""
    try:
        client = OpenSearch(
            hosts=[{
                "host": OPENSEARCH_HOST,
                "port": OPENSEARCH_PORT
            }],
            http_auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
            use_ssl=True,
            verify_certs=False,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True
        )
        
        # Test connection
        info = client.info()
        print(f"✅ Connected to OpenSearch: {info.get('cluster_name', 'Unknown')}")
        print(f"   Version: {info.get('version', {}).get('number', 'Unknown')}")
        print(f"   Host: {OPENSEARCH_HOST}:{OPENSEARCH_PORT}")
        print()
        
        return client
        
    except Exception as e:
        print(f"❌ Failed to connect to OpenSearch: {e}")
        print(f"💡 Check your connection details:")
        print(f"   Host: {OPENSEARCH_HOST}")
        print(f"   Port: {OPENSEARCH_PORT}")
        print(f"   User: {OPENSEARCH_USER}")
        print(f"   Password: {'Set' if OPENSEARCH_PASS else 'Not set'}")
        sys.exit(1)

def get_all_indices(client: OpenSearch) -> Dict[str, Any]:
    """Get all indices with their basic information"""
    try:
        indices = client.indices.get(index="*")
        return indices
    except Exception as e:
        print(f"❌ Failed to get indices: {e}")
        return {}

def get_index_stats(client: OpenSearch, index_name: str) -> Dict[str, Any]:
    """Get statistics for a specific index"""
    try:
        stats = client.indices.stats(index=index_name)
        return stats.get("indices", {}).get(index_name, {})
    except Exception as e:
        print(f"⚠️ Failed to get stats for {index_name}: {e}")
        return {}

def get_index_mapping(client: OpenSearch, index_name: str) -> Dict[str, Any]:
    """Get mapping for a specific index"""
    try:
        mapping = client.indices.get_mapping(index=index_name)
        return mapping.get(index_name, {}).get("mappings", {})
    except Exception as e:
        print(f"⚠️ Failed to get mapping for {index_name}: {e}")
        return {}

def format_size(size_bytes: int) -> str:
    """Format bytes to human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    size = float(size_bytes)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:.1f} {units[unit_index]}"

def inspect_opensearch_structure(client: OpenSearch):
    """Main inspection function"""
    print("🔍 OPENSEARCH STRUCTURE INSPECTION")
    print("=" * 60)
    
    # Get all indices
    all_indices = get_all_indices(client)
    
    if not all_indices:
        print("📭 No indices found in OpenSearch cluster")
        return
    
    # Separate system and user indices
    system_indices = []
    user_indices = []
    eval_indices = []
    
    for index_name in all_indices.keys():
        if index_name.startswith('.'):
            system_indices.append(index_name)
        elif index_name.startswith('eval-'):
            eval_indices.append(index_name)
        else:
            user_indices.append(index_name)
    
    print(f"📊 SUMMARY:")
    print(f"   Total indices: {len(all_indices)}")
    print(f"   🔧 System indices: {len(system_indices)}")
    print(f"   📋 Evaluation indices: {len(eval_indices)}")
    print(f"   📄 Other user indices: {len(user_indices)}")
    print()
    
    # Process evaluation indices (most important)
    if eval_indices:
        print("🎯 EVALUATION INDICES (Your Ask InnovAI Data):")
        print("-" * 50)
        
        total_eval_docs = 0
        total_eval_size = 0
        
        for index_name in sorted(eval_indices):
            stats = get_index_stats(client, index_name)
            mapping = get_index_mapping(client, index_name)
            
            # Extract key metrics
            doc_count = stats.get("primaries", {}).get("docs", {}).get("count", 0)
            store_size = stats.get("primaries", {}).get("store", {}).get("size_in_bytes", 0)
            
            total_eval_docs += doc_count
            total_eval_size += store_size
            
            print(f"📁 {index_name}")
            print(f"   📄 Documents: {doc_count:,}")
            print(f"   💾 Size: {format_size(store_size)}")
            
            # Check for key fields in mapping
            properties = mapping.get("properties", {})
            key_fields = []
            
            if "evaluationId" in properties:
                key_fields.append("evaluationId")
            if "template_id" in properties:
                key_fields.append("template_id")
            if "template_name" in properties:
                key_fields.append("template_name")
            if "chunks" in properties:
                key_fields.append(f"chunks ({properties['chunks'].get('type', 'unknown')})")
            if "full_text" in properties:
                key_fields.append("full_text")
            if "metadata" in properties:
                key_fields.append("metadata")
            
            if key_fields:
                print(f"   🏷️ Key fields: {', '.join(key_fields)}")
            
            # Check document structure version
            if "_structure_version" in properties:
                print(f"   🔄 Structure: Enhanced (v4.0+)")
            elif "chunks" in properties and properties["chunks"].get("type") == "nested":
                print(f"   🔄 Structure: Evaluation-grouped")
            else:
                print(f"   🔄 Structure: Legacy/Unknown")
            
            print()
        
        print(f"📊 EVALUATION TOTALS:")
        print(f"   📄 Total documents: {total_eval_docs:,}")
        print(f"   💾 Total size: {format_size(total_eval_size)}")
        print()
    
    # Process other user indices
    if user_indices:
        print("📄 OTHER USER INDICES:")
        print("-" * 30)
        
        for index_name in sorted(user_indices):
            stats = get_index_stats(client, index_name)
            doc_count = stats.get("primaries", {}).get("docs", {}).get("count", 0)
            store_size = stats.get("primaries", {}).get("store", {}).get("size_in_bytes", 0)
            
            print(f"📁 {index_name}")
            print(f"   📄 Documents: {doc_count:,}")
            print(f"   💾 Size: {format_size(store_size)}")
            print()
    
    # System indices summary (don't show details)
    if system_indices:
        print("🔧 SYSTEM INDICES:")
        print("-" * 20)
        print(f"   Found {len(system_indices)} system indices")
        print(f"   Examples: {', '.join(system_indices[:3])}")
        if len(system_indices) > 3:
            print(f"   ... and {len(system_indices) - 3} more")
        print("   (System indices are preserved during cleanup)")
        print()

def sample_documents(client: OpenSearch, index_name: str, count: int = 2):
    """Show sample documents from an index"""
    try:
        response = client.search(
            index=index_name,
            body={
                "size": count,
                "query": {"match_all": {}}
            }
        )
        
        hits = response.get("hits", {}).get("hits", [])
        
        if not hits:
            print(f"   📭 No documents found in {index_name}")
            return
        
        print(f"   📄 Sample documents from {index_name}:")
        
        for i, hit in enumerate(hits, 1):
            source = hit.get("_source", {})
            
            print(f"      {i}. Document ID: {hit.get('_id', 'Unknown')}")
            
            # Show key fields for evaluation documents
            if index_name.startswith('eval-'):
                eval_id = source.get("evaluationId", "Unknown")
                template_name = source.get("template_name", "Unknown")
                total_chunks = source.get("total_chunks", 0)
                
                print(f"         🆔 EvaluationID: {eval_id}")
                print(f"         📋 Template: {template_name}")
                print(f"         🧩 Chunks: {total_chunks}")
                
                # Show metadata if available
                metadata = source.get("metadata", {})
                if metadata:
                    agent = metadata.get("agent", "Unknown")
                    partner = metadata.get("partner", "Unknown")
                    site = metadata.get("site", "Unknown")
                    print(f"         👤 Agent: {agent} | 🤝 Partner: {partner} | 🏢 Site: {site}")
            else:
                # Show first few fields for other indices
                field_count = 0
                for key, value in source.items():
                    if field_count < 3:
                        if isinstance(value, (str, int, float, bool)):
                            print(f"         {key}: {value}")
                        else:
                            print(f"         {key}: {type(value).__name__}")
                        field_count += 1
                    else:
                        remaining_fields = len(source) - field_count
                        if remaining_fields > 0:
                            print(f"         ... and {remaining_fields} more fields")
                        break
            
            print()
    
    except Exception as e:
        print(f"   ⚠️ Failed to sample documents from {index_name}: {e}")

def delete_user_indices(client: OpenSearch):
    """Safely delete all user indices (preserves system indices)"""
    print("🗑️ DELETE USER INDICES")
    print("=" * 40)
    
    # Get all indices
    all_indices = get_all_indices(client)
    
    # Filter user indices (exclude system indices starting with '.')
    user_indices = [name for name in all_indices.keys() if not name.startswith('.')]
    
    if not user_indices:
        print("✅ No user indices found to delete")
        return
    
    print("⚠️ WARNING: This will DELETE the following indices:")
    print()
    
    total_docs = 0
    total_size = 0
    
    for index_name in sorted(user_indices):
        stats = get_index_stats(client, index_name)
        doc_count = stats.get("primaries", {}).get("docs", {}).get("count", 0)
        store_size = stats.get("primaries", {}).get("store", {}).get("size_in_bytes", 0)
        
        total_docs += doc_count
        total_size += store_size
        
        print(f"   🗑️ {index_name}")
        print(f"      📄 Documents: {doc_count:,}")
        print(f"      💾 Size: {format_size(store_size)}")
        print()
    
    print(f"📊 TOTAL TO DELETE:")
    print(f"   📁 Indices: {len(user_indices)}")
    print(f"   📄 Documents: {total_docs:,}")
    print(f"   💾 Data: {format_size(total_size)}")
    print()
    
    print("🛡️ System indices will be PRESERVED")
    print()
    
    # Double confirmation
    confirm1 = input("❓ Type 'DELETE' to confirm deletion: ").strip()
    if confirm1 != "DELETE":
        print("❌ Deletion cancelled")
        return
    
    confirm2 = input("❓ Type 'YES' to proceed with deletion: ").strip()
    if confirm2 != "YES":
        print("❌ Deletion cancelled")
        return
    
    print("\n🗑️ Starting deletion...")
    
    deleted_count = 0
    failed_count = 0
    
    for index_name in user_indices:
        try:
            print(f"   Deleting {index_name}...", end="")
            client.indices.delete(index=index_name)
            print(" ✅")
            deleted_count += 1
        except Exception as e:
            print(f" ❌ Error: {e}")
            failed_count += 1
    
    print(f"\n🎉 DELETION COMPLETE:")
    print(f"   ✅ Successfully deleted: {deleted_count} indices")
    if failed_count > 0:
        print(f"   ❌ Failed to delete: {failed_count} indices")
    print(f"   📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("🚀 Your OpenSearch cluster is now clean and ready for fresh imports!")

def main():
    """Main function with interactive menu"""
    print("🔍 OpenSearch Structure Inspector & Cleanup Tool")
    print("=" * 60)
    print(f"🔗 Connecting to: {OPENSEARCH_HOST}:{OPENSEARCH_PORT}")
    print()
    
    # Create client and connect
    client = create_opensearch_client()
    
    while True:
        print("📋 MENU:")
        print("   1. 🔍 Inspect OpenSearch structure")
        print("   2. 📄 Sample documents from indices")
        print("   3. 🗑️ Delete all user indices (DANGEROUS)")
        print("   4. 🚪 Exit")
        print()
        
        choice = input("Select option (1-4): ").strip()
        
        if choice == "1":
            print()
            inspect_opensearch_structure(client)
            
        elif choice == "2":
            print()
            all_indices = get_all_indices(client)
            user_indices = [name for name in all_indices.keys() if not name.startswith('.')]
            
            if not user_indices:
                print("📭 No user indices found")
            else:
                print("📁 Available indices:")
                for i, index_name in enumerate(sorted(user_indices), 1):
                    print(f"   {i}. {index_name}")
                print()
                
                try:
                    index_choice = int(input("Select index number (or 0 for all): ").strip())
                    
                    if index_choice == 0:
                        for index_name in sorted(user_indices):
                            sample_documents(client, index_name)
                    elif 1 <= index_choice <= len(user_indices):
                        selected_index = sorted(user_indices)[index_choice - 1]
                        sample_documents(client, selected_index, count=3)
                    else:
                        print("❌ Invalid selection")
                        
                except ValueError:
                    print("❌ Invalid input")
            
        elif choice == "3":
            print()
            delete_user_indices(client)
            
        elif choice == "4":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice, please select 1-4")
        
        print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    main()