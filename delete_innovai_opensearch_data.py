#!/usr/bin/env python3
"""
delete_opensearch_data.py - Simple OpenSearch Data Deletion Script
Version: 1.0.0 - Clean slate for enhanced structure migration

This script deletes ALL user data from OpenSearch to prepare for 
the new template_ID-based collections with evaluation grouping.

CAUTION: This is a DESTRUCTIVE operation that cannot be undone!
"""

import os
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("opensearch_cleanup")

def delete_all_opensearch_data():
    """
    Delete all user indices from OpenSearch
    DESTRUCTIVE OPERATION - USE WITH CAUTION
    """
    try:
        # Import OpenSearch client
        from opensearch_client import get_opensearch_manager
        
        print("🧹 OpenSearch Data Deletion Script")
        print("=" * 50)
        print("⚠️  WARNING: This will DELETE ALL your evaluation data!")
        print("⚠️  This operation CANNOT be undone!")
        print("=" * 50)
        
        # Test connection first
        manager = get_opensearch_manager()
        
        if not manager.test_connection():
            print("❌ Cannot connect to OpenSearch")
            print("   Check your OPENSEARCH_HOST and credentials")
            return False
        
        print(f"✅ Connected to OpenSearch: {manager.host}:{manager.port}")
        
        # Get all indices
        print("\n🔍 Scanning for indices...")
        try:
            all_indices = manager.client.indices.get(index="*")
        except Exception as e:
            print(f"❌ Failed to list indices: {e}")
            return False
        
        # Filter out system indices (those starting with '.')
        user_indices = []
        system_indices = []
        
        for index_name in all_indices.keys():
            if index_name.startswith('.'):
                system_indices.append(index_name)
            else:
                user_indices.append(index_name)
        
        print(f"📊 Found {len(all_indices)} total indices:")
        print(f"   📁 User indices: {len(user_indices)}")
        print(f"   ⚙️  System indices: {len(system_indices)} (will be preserved)")
        
        if not user_indices:
            print("\n✅ No user indices found to delete")
            print("   Your OpenSearch is already clean!")
            return True
        
        # Show what will be deleted
        print(f"\n📋 User indices that will be DELETED:")
        for i, index_name in enumerate(user_indices, 1):
            try:
                # Get document count for each index
                stats = manager.client.indices.stats(index=index_name)
                doc_count = stats["_all"]["primaries"]["docs"]["count"]
                store_size = stats["_all"]["primaries"]["store"]["size_in_bytes"]
                size_mb = store_size / (1024 * 1024) if store_size else 0
                
                print(f"   {i:2d}. {index_name}")
                print(f"       📄 Documents: {doc_count:,}")
                print(f"       💾 Size: {size_mb:.1f} MB")
                
            except Exception as e:
                print(f"   {i:2d}. {index_name} (error getting stats: {e})")
        
        # Calculate totals
        try:
            total_docs = 0
            total_size_mb = 0
            
            for index_name in user_indices:
                try:
                    stats = manager.client.indices.stats(index=index_name)
                    total_docs += stats["_all"]["primaries"]["docs"]["count"]
                    store_size = stats["_all"]["primaries"]["store"]["size_in_bytes"]
                    total_size_mb += store_size / (1024 * 1024) if store_size else 0
                except:
                    pass
            
            print(f"\n📊 TOTAL TO DELETE:")
            print(f"   📁 Indices: {len(user_indices)}")
            print(f"   📄 Documents: {total_docs:,}")
            print(f"   💾 Data: {total_size_mb:.1f} MB")
            
        except Exception as e:
            print(f"   ⚠️  Could not calculate totals: {e}")
        
        # Safety confirmation
        print(f"\n" + "=" * 50)
        print(f"🚨 FINAL CONFIRMATION REQUIRED")
        print(f"=" * 50)
        print(f"This will permanently delete {len(user_indices)} indices")
        print(f"and all their data from OpenSearch.")
        print(f"")
        print(f"Type 'DELETE ALL DATA' to confirm (case sensitive):")
        
        confirmation = input("Confirmation: ").strip()
        
        if confirmation != "DELETE ALL DATA":
            print("\n❌ Deletion cancelled - confirmation text did not match")
            print("   No data was deleted")
            return False
        
        # Perform deletion
        print(f"\n🗑️  Starting deletion process...")
        deleted_count = 0
        failed_count = 0
        
        for i, index_name in enumerate(user_indices, 1):
            try:
                print(f"   Deleting {i}/{len(user_indices)}: {index_name}...", end="")
                manager.client.indices.delete(index=index_name)
                print(" ✅")
                deleted_count += 1
                
            except Exception as e:
                print(f" ❌ Failed: {e}")
                failed_count += 1
        
        # Results summary
        print(f"\n" + "=" * 50)
        print(f"🎉 DELETION COMPLETE")
        print(f"=" * 50)
        print(f"✅ Successfully deleted: {deleted_count} indices")
        if failed_count > 0:
            print(f"❌ Failed to delete: {failed_count} indices")
        print(f"⚙️  System indices preserved: {len(system_indices)}")
        print(f"🕒 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if deleted_count == len(user_indices):
            print(f"\n🚀 OpenSearch is now clean and ready for the enhanced import!")
            print(f"   You can now run the new import with template_ID collections")
            print(f"   and evaluation grouping structure.")
            return True
        else:
            print(f"\n⚠️  Some indices could not be deleted")
            print(f"   You may need to delete them manually or check permissions")
            return False
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure opensearch_client.py is available")
        print("   and OpenSearch dependencies are installed")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.exception("Deletion failed with exception")
        return False

def main():
    """Main execution"""
    print("🛠️  Ask InnovAI - OpenSearch Data Deletion")
    print("   Preparing for enhanced structure migration")
    print("   Version: 1.0.0")
    print()
    
    # Check if user really wants to proceed
    print("⚠️  This script will delete ALL evaluation data from OpenSearch!")
    print("   Only proceed if you want to start fresh with the new structure.")
    print()
    
    proceed = input("Do you want to continue? (type 'yes' to proceed): ").strip().lower()
    
    if proceed != 'yes':
        print("❌ Operation cancelled")
        return
    
    # Run deletion
    success = delete_all_opensearch_data()
    
    if success:
        print(f"\n🎯 Next Steps:")
        print(f"   1. Update your app.py with the enhanced version")
        print(f"   2. Update opensearch_client.py with enhanced version")
        print(f"   3. Update chat_handlers.py with enhanced version")
        print(f"   4. Run the enhanced import process")
        print(f"   5. Verify the new structure with /test_enhanced_structure")
        
        print(f"\n📖 Benefits of new structure:")
        print(f"   ✅ Collections based on template_ID (cleaner)")
        print(f"   ✅ One document per evaluation (better aggregation)")
        print(f"   ✅ Grouped chunks (maintained detail)")
        print(f"   ✅ Better search results (evaluation-level)")
    else:
        print(f"\n❌ Deletion was not fully successful")
        print(f"   Check the logs above for specific errors")
        print(f"   You may need to manually clean some indices")

if __name__ == "__main__":
    main()