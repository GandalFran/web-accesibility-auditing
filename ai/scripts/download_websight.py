
import os
import csv
import socket
import time
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from datetime import datetime

# Increase default socket timeout
socket.setdefaulttimeout(120)

# Config
# Robustly resolve paths relative to this script: code/ai/scripts/ -> code/ai/data/
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"
METADATA_PATH = DATA_DIR / "scraping_metadata.csv"

# Download 2000 samples for development (Subset of 270 used for paper evaluation)
NUM_SAMPLES = 2000

# Setup
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading {NUM_SAMPLES} samples from HuggingFaceM4/WebSight (v0.2)...")
print(f"Output directory: {DATA_DIR}")

def get_dataset_stream():
    """Robustly get dataset stream with retries."""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Explicitly use v0.2 configuration as per paper
            return load_dataset("HuggingFaceM4/WebSight", "v0.2", split="train", streaming=True)
        except Exception as e:
            print(f"Error loading dataset (Attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("Failed to load dataset after retries")

# Load streaming dataset
ds = get_dataset_stream()

# CSV Writer
with open(METADATA_PATH, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['image_id', 'url', 'screenshot_path', 'timestamp', 'title', 'lang', 'num_images', 'num_headings']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Resilient iteration
    iterator = iter(ds)
    successful_samples = 0
    
    while successful_samples < NUM_SAMPLES:
        try:
            item = next(iterator)
            i = successful_samples
            
            # Filename
            filename = f"{i:05d}_websight_sample.png"
            filepath = SCREENSHOTS_DIR / filename
            
            # Save Image
            # WebSight v0.2 structure usually has 'image' key
            if 'image' in item:
                item['image'].save(filepath)
            else:
                print(f"Warning: Item {i} missing image key. Keys: {item.keys()}")
                continue
            
            # Metadata entry
            writer.writerow({
                'image_id': i,
                'url': f"http://websight-generated-{i}.com", # Synthetic URL
                'screenshot_path': str(filepath.absolute()),
                'timestamp': datetime.now().isoformat(),
                'title': f"WebSight Sample {i}",
                'lang': "en",
                'num_images': 0, # Placeholder
                'num_headings': 0 # Placeholder
            })
            
            successful_samples += 1
            if successful_samples % 10 == 0:
                print(f"Saved {successful_samples} samples...")
                
        except StopIteration:
            break
        except Exception as e:
            print(f"Warning: Error processing sample {successful_samples}, retrying... {e}")
            time.sleep(2)
            # Re-initialize iterator if needed or simply skip? 
            # Streaming iterator might be broken. We attempt to continue.
            continue

print("✓ Dataset download complete.")
