#!/usr/bin/env python3
"""
Git sync utilities for TrafCast data collection
"""

import os
import subprocess
from datetime import datetime

def git_commit_and_push_data(roads_updated, target_date, project_root=None):
    """
    Commit and push updated data files and metadata.

    Args:
        roads_updated: List of road keys that were updated
        target_date: Date that was collected
        project_root: Project root directory (auto-detected if None)
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    try:
        # Change to project directory
        original_cwd = os.getcwd()
        os.chdir(project_root)

        print(f"🔄 Syncing data updates to Git...")

        # Check if Git LFS is working
        try:
            subprocess.run(['git', 'lfs', 'version'], check=True,
                         capture_output=True, timeout=10)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️ Git LFS not found or not working")
            print("💡 Large files may not sync properly")

        # Add metadata file
        subprocess.run(['git', 'add', 'data/Los Angeles/roads/road_metadata.json'], check=True)
        print("✅ Added metadata file")

        # Add updated CSV files
        for road_key in roads_updated:
            csv_file = f"data/Los Angeles/roads/{road_key}.csv.gz"
            if os.path.exists(csv_file):
                subprocess.run(['git', 'add', csv_file], check=True)
                print(f"✅ Added {csv_file}")

        # Create commit message
        roads_list = ", ".join(roads_updated[:3])  # Show first 3 roads
        if len(roads_updated) > 3:
            roads_list += f" (+{len(roads_updated)-3} more)"

        commit_msg = f"📊 Data collection for {target_date}\n\nUpdated roads: {roads_list}\n\n🤖 Auto-committed by data collection script"

        # Commit changes
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
        print(f"✅ Committed changes: {len(roads_updated)} roads updated")

        # Push to remote (with LFS support)
        try:
            subprocess.run(['git', 'push'], check=True, timeout=300)  # 5 min timeout for LFS
            print("✅ Pushed to remote repository")
        except subprocess.TimeoutExpired:
            print("⚠️ Git push timed out (possibly large LFS files)")
            print("💡 Try manually: git push")
            return False

        return True

    except subprocess.CalledProcessError as e:
        error_msg = str(e)
        if 'git-lfs' in error_msg:
            print("❌ Git LFS error detected")
            print("💡 Fix: Ensure git-lfs is in PATH or configure LFS properly")
            print(f"   Error: {e}")
        else:
            print(f"❌ Git operation failed: {e}")
        return False

    except Exception as e:
        print(f"❌ Error during git sync: {e}")
        return False

    finally:
        # Restore original directory
        os.chdir(original_cwd)

def check_git_status(project_root=None):
    """Check if there are uncommitted changes."""
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    try:
        original_cwd = os.getcwd()
        os.chdir(project_root)

        result = subprocess.run(['git', 'status', '--porcelain'],
                              capture_output=True, text=True, check=True)

        if result.stdout.strip():
            print("📋 Uncommitted changes found:")
            print(result.stdout)
            return False
        else:
            print("✅ Git working directory is clean")
            return True

    except Exception as e:
        print(f"⚠️ Could not check git status: {e}")
        return None

    finally:
        os.chdir(original_cwd)