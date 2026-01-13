#!/usr/bin/env python3
"""
Partner & Site Field Diagnostic Script
Checks the actual structure and values in OpenSearch
"""

import sys
import os
sys.path.insert(0, '/mnt/project')

from opensearch_client import get_opensearch_client, test_connection
import json

def diagnose_partner_site_fields():
    """Comprehensive diagnosis of partner and site field issues"""
    
    print("=" * 80)
    print("PARTNER & SITE FIELD DIAGNOSTIC")
    print("=" * 80)
    
    # Test connection
    print("\n1. Testing OpenSearch connection...")
    if not test_connection():
        print("❌ OpenSearch not available")
        return
    print("✅ OpenSearch connected")
    
    client = get_opensearch_client()
    
    # Get sample documents
    print("\n2. Fetching sample documents...")
    try:
        response = client.search(
            index="eval-*",
            body={
                "query": {"match_all": {}},
                "size": 5,
                "_source": True
            }
        )
        
        hits = response.get("hits", {}).get("hits", [])
        print(f"✅ Retrieved {len(hits)} sample documents")
        
    except Exception as e:
        print(f"❌ Failed to fetch documents: {e}")
        return
    
    # Analyze field structure
    print("\n3. Analyzing field structure...")
    print("-" * 80)
    
    field_structures = {
        "top_level_partner": 0,
        "metadata_partner": 0,
        "top_level_site": 0,
        "metadata_site": 0,
        "missing_partner": 0,
        "missing_site": 0
    }
    
    partner_values = set()
    site_values = set()
    
    for i, hit in enumerate(hits):
        source = hit.get("_source", {})
        eval_id = source.get("evaluationId", "unknown")
        
        print(f"\nDocument {i+1} (evaluationId: {eval_id}):")
        print(f"  Index: {hit.get('_index')}")
        
        # Check top-level fields
        has_top_partner = "partner" in source
        has_top_site = "site" in source
        
        # Check metadata fields
        metadata = source.get("metadata", {})
        has_meta_partner = isinstance(metadata, dict) and "partner" in metadata
        has_meta_site = isinstance(metadata, dict) and "site" in metadata
        
        # Update counters
        if has_top_partner:
            field_structures["top_level_partner"] += 1
            partner_val = source.get("partner")
            if partner_val:
                partner_values.add(partner_val)
                print(f"  ✅ Top-level partner: '{partner_val}'")
        
        if has_meta_partner:
            field_structures["metadata_partner"] += 1
            partner_val = metadata.get("partner")
            if partner_val:
                partner_values.add(partner_val)
                print(f"  ✅ Metadata partner: '{partner_val}'")
        
        if has_top_site:
            field_structures["top_level_site"] += 1
            site_val = source.get("site")
            if site_val:
                site_values.add(site_val)
                print(f"  ✅ Top-level site: '{site_val}'")
        
        if has_meta_site:
            field_structures["metadata_site"] += 1
            site_val = metadata.get("site")
            if site_val:
                site_values.add(site_val)
                print(f"  ✅ Metadata site: '{site_val}'")
        
        if not has_top_partner and not has_meta_partner:
            field_structures["missing_partner"] += 1
            print(f"  ❌ No partner field found")
        
        if not has_top_site and not has_meta_site:
            field_structures["missing_site"] += 1
            print(f"  ❌ No site field found")
    
    # Summary
    print("\n" + "=" * 80)
    print("FIELD STRUCTURE SUMMARY")
    print("=" * 80)
    print(f"\nTotal documents analyzed: {len(hits)}")
    print(f"\nPartner field locations:")
    print(f"  Top-level (partner): {field_structures['top_level_partner']}")
    print(f"  Metadata (metadata.partner): {field_structures['metadata_partner']}")
    print(f"  Missing: {field_structures['missing_partner']}")
    
    print(f"\nSite field locations:")
    print(f"  Top-level (site): {field_structures['top_level_site']}")
    print(f"  Metadata (metadata.site): {field_structures['metadata_site']}")
    print(f"  Missing: {field_structures['missing_site']}")
    
    print(f"\nUnique partner values found: {len(partner_values)}")
    if partner_values:
        for val in sorted(partner_values):
            print(f"  - {val}")
    
    print(f"\nUnique site values found: {len(site_values)}")
    if site_values:
        for val in sorted(site_values):
            print(f"  - {val}")
    
    # Test aggregations
    print("\n" + "=" * 80)
    print("AGGREGATION TESTS")
    print("=" * 80)
    
    # Test different field paths for aggregations
    agg_tests = [
        ("partner.keyword", "Top-level partner (exact)"),
        ("metadata.partner.keyword", "Metadata partner (exact)"),
        ("site.keyword", "Top-level site (exact)"),
        ("metadata.site.keyword", "Metadata site (exact)"),
    ]
    
    for field_path, description in agg_tests:
        print(f"\n{description} ({field_path}):")
        try:
            agg_response = client.search(
                index="eval-*",
                body={
                    "size": 0,
                    "aggs": {
                        "values": {
                            "terms": {
                                "field": field_path,
                                "size": 10
                            }
                        }
                    }
                }
            )
            
            buckets = agg_response.get("aggregations", {}).get("values", {}).get("buckets", [])
            
            if buckets:
                print(f"  ✅ Found {len(buckets)} distinct values:")
                for bucket in buckets[:5]:  # Show first 5
                    print(f"     - {bucket['key']}: {bucket['doc_count']} documents")
            else:
                print(f"  ⚠️  No values found (field might not exist or be empty)")
                
        except Exception as e:
            print(f"  ❌ Aggregation failed: {str(e)}")
    
    # Check index mapping
    print("\n" + "=" * 80)
    print("INDEX MAPPING CHECK")
    print("=" * 80)
    
    try:
        # Get mapping for one index
        indices = client.cat.indices(index="eval-*", format="json")
        if indices:
            first_index = indices[0]["index"]
            print(f"\nChecking mapping for: {first_index}")
            
            mapping = client.indices.get_mapping(index=first_index)
            properties = mapping.get(first_index, {}).get("mappings", {}).get("properties", {})
            
            # Check top-level partner
            if "partner" in properties:
                print(f"\n✅ Top-level 'partner' field mapping:")
                print(f"   {json.dumps(properties['partner'], indent=2)}")
            else:
                print(f"\n❌ No top-level 'partner' field in mapping")
            
            # Check top-level site
            if "site" in properties:
                print(f"\n✅ Top-level 'site' field mapping:")
                print(f"   {json.dumps(properties['site'], indent=2)}")
            else:
                print(f"\n❌ No top-level 'site' field in mapping")
            
            # Check metadata.partner
            if "metadata" in properties:
                metadata_props = properties["metadata"].get("properties", {})
                if "partner" in metadata_props:
                    print(f"\n✅ 'metadata.partner' field mapping:")
                    print(f"   {json.dumps(metadata_props['partner'], indent=2)}")
                else:
                    print(f"\n❌ No 'metadata.partner' field in mapping")
                
                if "site" in metadata_props:
                    print(f"\n✅ 'metadata.site' field mapping:")
                    print(f"   {json.dumps(metadata_props['site'], indent=2)}")
                else:
                    print(f"\n❌ No 'metadata.site' field in mapping")
            else:
                print(f"\n❌ No 'metadata' object in mapping")
                
    except Exception as e:
        print(f"❌ Failed to get mapping: {e}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if field_structures["metadata_partner"] > 0 and field_structures["top_level_partner"] == 0:
        print("\n✅ Data is correctly structured with metadata.partner")
        print("   Search queries should use: metadata.partner.keyword")
    elif field_structures["top_level_partner"] > 0 and field_structures["metadata_partner"] == 0:
        print("\n⚠️  Data has top-level partner field")
        print("   Search queries should use: partner.keyword")
    elif field_structures["metadata_partner"] > 0 and field_structures["top_level_partner"] > 0:
        print("\n⚠️  MIXED STRUCTURE: Both top-level and metadata partner fields exist")
        print("   This can cause inconsistent results!")
        print("   Consider re-indexing to use only one structure")
    
    if field_structures["missing_partner"] > 0:
        print(f"\n⚠️  {field_structures['missing_partner']} documents missing partner field")
    
    if field_structures["missing_site"] > 0:
        print(f"\n⚠️  {field_structures['missing_site']} documents missing site field")

if __name__ == "__main__":
    diagnose_partner_site_fields()
