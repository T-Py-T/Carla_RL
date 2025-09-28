#!/usr/bin/env python3
"""
CLI tool for content-addressable storage management.

Provides command-line interface for managing content-addressable storage,
viewing statistics, and performing maintenance operations.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from versioning.content_storage import ContentAddressableStorage, ContentStorageError
from versioning.artifact_manager import ArtifactManager
from versioning import ContentAddressableArtifactManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_content_storage(storage_dir: str) -> ContentAddressableStorage:
    """Set up content-addressable storage with specified directory."""
    storage_path = Path(storage_dir)
    if not storage_path.exists():
        logger.error(f"Storage directory does not exist: {storage_dir}")
        sys.exit(1)
    
    return ContentAddressableStorage(storage_path)


def setup_artifact_manager(artifacts_dir: str, content_storage_dir: Optional[str] = None) -> ContentAddressableArtifactManager:
    """Set up content-addressable artifact manager."""
    artifacts_path = Path(artifacts_dir)
    if not artifacts_path.exists():
        logger.error(f"Artifacts directory does not exist: {artifacts_dir}")
        sys.exit(1)
    
    content_storage_path = Path(content_storage_dir) if content_storage_dir else None
    return ContentAddressableArtifactManager(artifacts_path, content_storage_path)


def store_content(storage: ContentAddressableStorage, 
                 content_path: str,
                 metadata: Optional[dict] = None,
                 json_output: bool = False) -> None:
    """Store content in content-addressable storage."""
    try:
        content_file = Path(content_path)
        if not content_file.exists():
            logger.error(f"Content file does not exist: {content_path}")
            sys.exit(1)
        
        content_hash = storage.store_content(content_file, metadata)
        
        if json_output:
            output = {
                "content_hash": content_hash,
                "file_path": str(content_file),
                "size": content_file.stat().st_size,
                "metadata": metadata or {}
            }
            print(json.dumps(output, indent=2))
        else:
            print(f"Stored content: {content_hash}")
            print(f"  File: {content_file}")
            print(f"  Size: {content_file.stat().st_size:,} bytes")
            if metadata:
                print(f"  Metadata: {metadata}")
    
    except ContentStorageError as e:
        logger.error(f"Failed to store content: {e}")
        sys.exit(1)


def retrieve_content(storage: ContentAddressableStorage,
                    content_hash: str,
                    output_path: Optional[str] = None,
                    json_output: bool = False) -> None:
    """Retrieve content from storage."""
    try:
        if output_path:
            # Copy to file
            target_path = Path(output_path)
            storage.copy_content_to(content_hash, target_path)
            
            if json_output:
                output = {
                    "content_hash": content_hash,
                    "output_path": str(target_path),
                    "size": target_path.stat().st_size
                }
                print(json.dumps(output, indent=2))
            else:
                print(f"Retrieved content: {content_hash}")
                print(f"  Output: {target_path}")
                print(f"  Size: {target_path.stat().st_size:,} bytes")
        else:
            # Print to stdout
            content = storage.retrieve_content(content_hash)
            
            if json_output:
                output = {
                    "content_hash": content_hash,
                    "size": len(content),
                    "content": content.decode('utf-8', errors='ignore')
                }
                print(json.dumps(output, indent=2))
            else:
                print(content.decode('utf-8', errors='ignore'))
    
    except ContentStorageError as e:
        logger.error(f"Failed to retrieve content: {e}")
        sys.exit(1)


def check_content_exists(storage: ContentAddressableStorage,
                        content_hash: str,
                        json_output: bool = False) -> None:
    """Check if content exists in storage."""
    exists = storage.content_exists(content_hash)
    
    if json_output:
        output = {
            "content_hash": content_hash,
            "exists": exists
        }
        print(json.dumps(output, indent=2))
    else:
        if exists:
            print(f"Content exists: {content_hash}")
        else:
            print(f"Content not found: {content_hash}")
            sys.exit(1)


def get_content_info(storage: ContentAddressableStorage,
                    content_hash: str,
                    json_output: bool = False) -> None:
    """Get content information."""
    info = storage.get_content_info(content_hash)
    
    if not info:
        print(f"Content not found: {content_hash}")
        sys.exit(1)
    
    if json_output:
        print(json.dumps(info.to_dict(), indent=2))
    else:
        print(f"Content Information: {content_hash}")
        print(f"  Size: {info.size:,} bytes")
        print(f"  Created: {info.created_at}")
        print(f"  Last Accessed: {info.last_accessed}")
        print(f"  Access Count: {info.access_count}")
        if info.metadata:
            print(f"  Metadata: {info.metadata}")


def list_content(storage: ContentAddressableStorage,
                limit: Optional[int] = None,
                json_output: bool = False) -> None:
    """List all content in storage."""
    content_list = storage.list_content(limit)
    
    if not content_list:
        print("No content found in storage")
        return
    
    if json_output:
        output = [ref.to_dict() for ref in content_list]
        print(json.dumps(output, indent=2))
    else:
        print(f"Content in Storage ({len(content_list)} items):")
        print("-" * 80)
        for ref in content_list:
            print(f"{ref.content_hash}  {ref.size:>10,} bytes  {ref.created_at}  (accessed {ref.access_count} times)")


def delete_content(storage: ContentAddressableStorage,
                  content_hash: str,
                  json_output: bool = False) -> None:
    """Delete content from storage."""
    try:
        success = storage.delete_content(content_hash)
        
        if json_output:
            output = {
                "content_hash": content_hash,
                "deleted": success
            }
            print(json.dumps(output, indent=2))
        else:
            if success:
                print(f"Deleted content: {content_hash}")
            else:
                print(f"Content not found: {content_hash}")
                sys.exit(1)
    
    except ContentStorageError as e:
        logger.error(f"Failed to delete content: {e}")
        sys.exit(1)


def show_storage_stats(storage: ContentAddressableStorage,
                      json_output: bool = False) -> None:
    """Show storage statistics."""
    stats = storage.get_storage_stats()
    
    if json_output:
        print(json.dumps({
            "total_objects": stats.total_objects,
            "total_size": stats.total_size,
            "unique_hashes": stats.unique_hashes,
            "duplicate_objects": stats.duplicate_objects,
            "storage_efficiency": stats.storage_efficiency,
            "oldest_object": stats.oldest_object,
            "newest_object": stats.newest_object,
            "most_accessed": stats.most_accessed,
            "least_accessed": stats.least_accessed
        }, indent=2))
    else:
        print("Storage Statistics:")
        print("-" * 40)
        print(f"Total Objects: {stats.total_objects:,}")
        print(f"Total Size: {stats.total_size:,} bytes ({stats.total_size / (1024**3):.2f} GB)")
        print(f"Unique Hashes: {stats.unique_hashes:,}")
        print(f"Duplicate Objects: {stats.duplicate_objects:,}")
        print(f"Storage Efficiency: {stats.storage_efficiency:.1f}%")
        
        if stats.oldest_object:
            print(f"Oldest Object: {stats.oldest_object}")
        if stats.newest_object:
            print(f"Newest Object: {stats.newest_object}")
        if stats.most_accessed:
            print(f"Most Accessed: {stats.most_accessed}")
        if stats.least_accessed:
            print(f"Least Accessed: {stats.least_accessed}")


def cleanup_orphaned(storage: ContentAddressableStorage,
                    json_output: bool = False) -> None:
    """Clean up orphaned content."""
    try:
        orphaned_count = storage.cleanup_orphaned_content()
        
        if json_output:
            output = {
                "orphaned_files_cleaned": orphaned_count
            }
            print(json.dumps(output, indent=2))
        else:
            print(f"Cleaned up {orphaned_count} orphaned files")
    
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        sys.exit(1)


def verify_integrity(storage: ContentAddressableStorage,
                    json_output: bool = False) -> None:
    """Verify integrity of all content."""
    try:
        results = storage.verify_integrity()
        
        total_objects = len(results)
        valid_objects = sum(results.values())
        invalid_objects = total_objects - valid_objects
        
        if json_output:
            output = {
                "total_objects": total_objects,
                "valid_objects": valid_objects,
                "invalid_objects": invalid_objects,
                "integrity_percentage": (valid_objects / total_objects * 100) if total_objects > 0 else 0,
                "results": results
            }
            print(json.dumps(output, indent=2))
        else:
            print("Integrity Verification Results:")
            print("-" * 40)
            print(f"Total Objects: {total_objects:,}")
            print(f"Valid Objects: {valid_objects:,}")
            print(f"Invalid Objects: {invalid_objects:,}")
            print(f"Integrity: {(valid_objects / total_objects * 100):.1f}%" if total_objects > 0 else "N/A")
            
            if invalid_objects > 0:
                print("\nInvalid Objects:")
                for content_hash, is_valid in results.items():
                    if not is_valid:
                        print(f"  {content_hash}")
    
    except Exception as e:
        logger.error(f"Integrity verification failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Content-addressable storage management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Store content
  python content_storage_manager.py --storage-dir /path/to/storage store --content /path/to/file.txt
  
  # Retrieve content
  python content_storage_manager.py --storage-dir /path/to/storage retrieve --hash abc123 --output /path/to/output.txt
  
  # List all content
  python content_storage_manager.py --storage-dir /path/to/storage list
  
  # Show storage statistics
  python content_storage_manager.py --storage-dir /path/to/storage stats
  
  # Verify integrity
  python content_storage_manager.py --storage-dir /path/to/storage verify
        """
    )
    
    parser.add_argument(
        "--storage-dir",
        required=True,
        help="Path to content storage directory"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Store command
    store_parser = subparsers.add_parser("store", help="Store content")
    store_parser.add_argument(
        "--content",
        required=True,
        help="Path to content file"
    )
    store_parser.add_argument(
        "--metadata",
        help="JSON metadata string"
    )
    
    # Retrieve command
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve content")
    retrieve_parser.add_argument(
        "--hash",
        required=True,
        help="Content hash"
    )
    retrieve_parser.add_argument(
        "--output",
        help="Output file path (if not provided, prints to stdout)"
    )
    
    # Check command
    check_parser = subparsers.add_parser("check", help="Check if content exists")
    check_parser.add_argument(
        "--hash",
        required=True,
        help="Content hash"
    )
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get content information")
    info_parser.add_argument(
        "--hash",
        required=True,
        help="Content hash"
    )
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all content")
    list_parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of items"
    )
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete content")
    delete_parser.add_argument(
        "--hash",
        required=True,
        help="Content hash"
    )
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show storage statistics")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up orphaned content")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify content integrity")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Set up content storage
    try:
        storage = setup_content_storage(args.storage_dir)
    except Exception as e:
        logger.error(f"Failed to set up content storage: {e}")
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == "store":
            metadata = None
            if args.metadata:
                metadata = json.loads(args.metadata)
            store_content(storage, args.content, metadata, args.json)
        
        elif args.command == "retrieve":
            retrieve_content(storage, args.hash, args.output, args.json)
        
        elif args.command == "check":
            check_content_exists(storage, args.hash, args.json)
        
        elif args.command == "info":
            get_content_info(storage, args.hash, args.json)
        
        elif args.command == "list":
            list_content(storage, args.limit, args.json)
        
        elif args.command == "delete":
            delete_content(storage, args.hash, args.json)
        
        elif args.command == "stats":
            show_storage_stats(storage, args.json)
        
        elif args.command == "cleanup":
            cleanup_orphaned(storage, args.json)
        
        elif args.command == "verify":
            verify_integrity(storage, args.json)
    
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
