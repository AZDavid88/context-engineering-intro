#!/usr/bin/env python3
"""
Test File Migration Utility

Systematically migrate scattered root-level test files to proper /tests/ structure.
Categorizes tests by type (unit, integration, research) and target module.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

def create_test_directories():
    """Create proper test directory structure."""
    test_dirs = [
        'tests/unit',
        'tests/integration', 
        'tests/research_archive',
        'tests/system'
    ]
    
    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def get_migration_plan() -> Dict[str, Tuple[str, str]]:
    """
    Define migration mapping: old_file -> (new_location, category)
    
    Categories:
    - unit: Unit tests for individual components
    - integration: Integration tests for component interactions
    - research_archive: Research/development artifacts to archive
    - system: System-level end-to-end tests
    """
    
    migration_map = {
        # Unit tests - test individual components
        'test_standalone_config.py': ('tests/unit/test_config.py', 'unit'),
        'test_standalone_data_storage.py': ('tests/unit/test_data_storage.py', 'unit'),
        'test_standalone_data_pipeline.py': ('tests/unit/test_data_pipeline.py', 'unit'),
        'test_hyperliquid_assets.py': ('tests/unit/test_hyperliquid_client.py', 'unit'),
        'test_data_ingestion.py': ('tests/unit/test_data_ingestion.py', 'unit'),
        'test_asset_universe_filter.py': ('tests/unit/test_asset_filter.py', 'unit'),
        'test_enhanced_rate_limiting_validation.py': ('tests/unit/test_rate_limiting.py', 'unit'),
        
        # Integration tests - test component interactions  
        'test_asset_to_ga_flow.py': ('tests/integration/test_asset_flow.py', 'integration'),
        'test_genetic_runs.py': ('tests/integration/test_genetic_engine.py', 'integration'),
        'test_monitoring_integration.py': ('tests/integration/test_monitoring.py', 'integration'),
        'test_session_management_validation.py': ('tests/integration/test_session_mgmt.py', 'integration'),
        'test_hierarchical_discovery_e2e.py': ('tests/system/test_hierarchical_discovery.py', 'system'),
        
        # Research artifacts - archive for historical reference
        'test_data_storage_simple.py': ('tests/research_archive/test_data_storage_simple.py', 'research'),
        'test_data_storage_research_driven.py': ('tests/research_archive/test_data_storage_research.py', 'research'),
        'test_strategy_engine_research.py': ('tests/research_archive/test_strategy_engine_research.py', 'research'),
        'test_position_sizer_research.py': ('tests/research_archive/test_position_sizer_research.py', 'research'),
        'test_order_management_research.py': ('tests/research_archive/test_order_management_research.py', 'research'),
        'test_phase_4b_optimization.py': ('tests/research_archive/test_phase_4b_optimization.py', 'research'),
    }
    
    return migration_map

def migrate_test_files(dry_run: bool = True):
    """
    Migrate test files according to the migration plan.
    
    Args:
        dry_run: If True, only show what would be moved without actually moving
    """
    
    migration_plan = get_migration_plan()
    
    print(f"🔄 Test File Migration {'(DRY RUN)' if dry_run else '(EXECUTING)'}")
    print("=" * 60)
    
    # Create directories first
    if not dry_run:
        create_test_directories()
    
    moved_count = 0
    missing_count = 0
    
    for old_file, (new_location, category) in migration_plan.items():
        if os.path.exists(old_file):
            print(f"📁 {category.upper()}: {old_file} → {new_location}")
            
            if not dry_run:
                # Create parent directory if it doesn't exist
                parent_dir = os.path.dirname(new_location)
                os.makedirs(parent_dir, exist_ok=True)
                
                # Move the file
                shutil.move(old_file, new_location)
                print(f"   ✅ Moved successfully")
                
            moved_count += 1
        else:
            print(f"   ⚠️ File not found: {old_file}")
            missing_count += 1
    
    print("\n" + "=" * 60)
    print(f"📊 MIGRATION SUMMARY:")
    print(f"   Files to migrate: {moved_count}")
    print(f"   Files missing: {missing_count}")
    print(f"   Total planned: {len(migration_plan)}")
    
    if dry_run:
        print(f"\n🔄 This was a dry run. Use migrate_test_files(dry_run=False) to execute.")
    else:
        print(f"\n✅ Migration completed!")
        
        # Validate migration
        print(f"\n🔍 VALIDATION:")
        for category in ['unit', 'integration', 'research_archive', 'system']:
            test_dir = f'tests/{category}'
            if os.path.exists(test_dir):
                file_count = len([f for f in os.listdir(test_dir) if f.endswith('.py')])
                print(f"   {category}: {file_count} test files")

def main():
    """Run test file migration."""
    print("🧪 Test File Migration Utility")
    print("=" * 40)
    
    # First, show the plan
    print("\n📋 MIGRATION PLAN:")
    migration_plan = get_migration_plan()
    
    categories = {}
    for old_file, (new_location, category) in migration_plan.items():
        if category not in categories:
            categories[category] = []
        categories[category].append((old_file, new_location))
    
    for category, files in categories.items():
        print(f"\n🏷️ {category.upper()} ({len(files)} files):")
        for old_file, new_location in files:
            status = "✅" if os.path.exists(old_file) else "❌"
            print(f"   {status} {old_file} → {new_location}")
    
    # Execute dry run first
    print(f"\n" + "=" * 60)
    migrate_test_files(dry_run=True)
    
    # Ask for confirmation (in interactive mode)
    try:
        response = input(f"\n❓ Proceed with migration? (y/N): ").lower().strip()
        if response == 'y':
            migrate_test_files(dry_run=False)
        else:
            print("❌ Migration cancelled.")
    except:
        # Non-interactive mode - just run the migration
        print("🤖 Non-interactive mode detected - executing migration...")
        migrate_test_files(dry_run=False)

if __name__ == "__main__":
    main()