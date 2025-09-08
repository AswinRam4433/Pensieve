#!/usr/bin/env python3
"""
Index Management Utility for FAISS Image Search Pipeline

This script provides command-line tools to manage your FAISS index:
- View index information
- Clear the index
- Add new images from URLs
- Search for similar images
- Backup and restore indexes

Usage:
    python index_manager.py --info                    # Show index information
    python index_manager.py --clear                   # Clear the index
    python index_manager.py --add-url <url>          # Add single image
    python index_manager.py --add-urls <file.txt>    # Add images from URL list
    python index_manager.py --search <image_url>     # Search for similar images
    python index_manager.py --backup <backup_name>   # Create backup
    python index_manager.py --restore <backup_name>  # Restore from backup
"""

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO

# Import your pipeline (make sure pipeline.py is in the same directory)
try:
    from pipeline import Pipeline
except ImportError:
    print("Error: Could not import Pipeline class. Make sure pipeline.py is in the same directory.")
    exit(1)


def show_info(pipeline):
    """Display current index information"""
    info = pipeline.get_index_info()
    print("\n=== INDEX INFORMATION ===")
    print(f"Total vectors: {info['total_vectors']}")
    print(f"Vector dimension: {info['dimension']}")
    print(f"Metadata entries: {info['metadata_entries']}")
    print(f"Index file: {info['index_path']}")
    print(f"Metadata file: {info['metadata_path']}")
    
    # Show some sample metadata
    if pipeline.image_metadata:
        print(f"\nSample metadata (first 3 entries):")
        for i, metadata in enumerate(pipeline.image_metadata[:3]):
            print(f"  {i+1}. URL: {metadata.get('url', 'N/A')}")
            print(f"     Index ID: {metadata.get('index_id', 'N/A')}")
            print(f"     Added at: {metadata.get('added_at', 'N/A')}")


def clear_index(pipeline):
    """Clear the entire index"""
    confirm = input("Are you sure you want to clear the entire index? (y/N): ")
    if confirm.lower() == 'y':
        pipeline.clear_index()
        pipeline.save_index()
        print("Index cleared and saved.")
    else:
        print("Operation cancelled.")


def add_single_url(pipeline, url):
    """Add a single image from URL"""
    try:
        print(f"Adding image from: {url}")
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content)).convert("RGB")
        embedding = pipeline.embed_img(image)
        
        metadata = {
            'url': url,
            'added_at': str(datetime.now()),
            'source': 'manual_add'
        }
        
        pipeline.add_vector(embedding, metadata)
        pipeline.save_index()
        print("Image added successfully!")
        
    except Exception as e:
        print(f"Error adding image: {e}")


def add_urls_from_file(pipeline, file_path):
    """Add multiple images from a file containing URLs"""
    try:
        with open(file_path, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(urls)} URLs in {file_path}")
        confirm = input("Do you want to proceed? (y/N): ")
        
        if confirm.lower() == 'y':
            added = pipeline.add_images_from_urls(urls, batch_size=10)
            print(f"Successfully added {added}/{len(urls)} images")
        else:
            print("Operation cancelled.")
            
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
    except Exception as e:
        print(f"Error reading file: {e}")


def search_similar(pipeline, query_url, top_k=5):
    """Search for similar images"""
    try:
        print(f"Searching for images similar to: {query_url}")
        
        response = requests.get(query_url, stream=True, timeout=10)
        response.raise_for_status()
        
        query_image = Image.open(BytesIO(response.content)).convert("RGB")
        results = pipeline.img_search(query_image, top_k=top_k)
        
        print(f"\nTop {len(results)} similar images:")
        for i, result in enumerate(results):
            print(f"\n{i+1}. Distance: {result['distance']:.4f}")
            print(f"   URL: {result['metadata'].get('url', 'N/A')}")
            print(f"   Added: {result['metadata'].get('added_at', 'N/A')}")
            print(f"   Index: {result['index']}")
            
    except Exception as e:
        print(f"Error searching: {e}")


def backup_index(pipeline, backup_name):
    """Create a backup of the current index"""
    try:
        backup_dir = Path("backups")
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{backup_name}_{timestamp}"
        backup_path.mkdir()
        
        # Copy index and metadata files
        shutil.copy2(pipeline.index_path, backup_path / "index.faiss")
        shutil.copy2(pipeline.metadata_path, backup_path / "metadata.json")
        
        print(f"Backup created at: {backup_path}")
        
    except Exception as e:
        print(f"Error creating backup: {e}")


def restore_index(pipeline, backup_name):
    """Restore index from backup"""
    try:
        backup_dir = Path("backups")
        
        # Find the most recent backup with the given name
        backups = list(backup_dir.glob(f"{backup_name}_*"))
        if not backups:
            print(f"No backups found with name: {backup_name}")
            return
        
        latest_backup = max(backups)
        print(f"Restoring from: {latest_backup}")
        
        confirm = input("This will overwrite the current index. Continue? (y/N): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            return
        
        # Copy backup files to current location
        shutil.copy2(latest_backup / "index.faiss", pipeline.index_path)
        shutil.copy2(latest_backup / "metadata.json", pipeline.metadata_path)
        
        # Reload the pipeline
        pipeline.load_index()
        print("Index restored successfully!")
        
    except Exception as e:
        print(f"Error restoring backup: {e}")


def list_backups():
    """List available backups"""
    backup_dir = Path("backups")
    if not backup_dir.exists():
        print("No backups directory found.")
        return
    
    backups = list(backup_dir.glob("*"))
    if not backups:
        print("No backups found.")
        return
    
    print("\nAvailable backups:")
    for backup in sorted(backups):
        print(f"  {backup.name}")


def main():
    parser = argparse.ArgumentParser(description="FAISS Index Management Utility")
    parser.add_argument("--info", action="store_true", help="Show index information")
    parser.add_argument("--clear", action="store_true", help="Clear the entire index")
    parser.add_argument("--add-url", help="Add a single image from URL")
    parser.add_argument("--add-urls", help="Add images from a file containing URLs")
    parser.add_argument("--search", help="Search for similar images using a query URL")
    parser.add_argument("--top-k", type=int, default=5, help="Number of similar images to return (default: 5)")
    parser.add_argument("--backup", help="Create a backup with the given name")
    parser.add_argument("--restore", help="Restore from backup with the given name")
    parser.add_argument("--list-backups", action="store_true", help="List available backups")
    
    args = parser.parse_args()
    
    # Handle list-backups without initializing pipeline
    if args.list_backups:
        list_backups()
        return
    
    # Initialize pipeline for other operations
    print("Initializing pipeline...")
    pipeline = Pipeline()
    
    if args.info:
        show_info(pipeline)
    elif args.clear:
        clear_index(pipeline)
    elif args.add_url:
        add_single_url(pipeline, args.add_url)
    elif args.add_urls:
        add_urls_from_file(pipeline, args.add_urls)
    elif args.search:
        search_similar(pipeline, args.search, args.top_k)
    elif args.backup:
        backup_index(pipeline, args.backup)
    elif args.restore:
        restore_index(pipeline, args.restore)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()