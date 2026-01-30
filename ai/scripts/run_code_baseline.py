
import csv
import re
import socket
import time
from pathlib import Path
from datasets import load_dataset
from bs4 import BeautifulSoup
import pandas as pd

# Increase default socket timeout
socket.setdefaulttimeout(120)

# Config
DATA_DIR = Path("data")
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / "code_baseline_results.csv"
NUM_SAMPLES = 2000

print(f"Running Code-Based Audit on {NUM_SAMPLES} samples...")

def get_dataset_stream():
    """Robustly get dataset stream with retries."""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            return load_dataset("HuggingFaceM4/WebSight", split="train", streaming=True)
        except Exception as e:
            print(f"Error loading dataset (Attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("Failed to load dataset after retries")

# Load streaming dataset
ds = get_dataset_stream()

results = []
iterator = iter(ds)
successful_samples = 0

print("Starting robust iteration...")

while successful_samples < NUM_SAMPLES:
    try:
        # 1. Fetch Data
        item = next(iterator)
        i = successful_samples
        html_content = item['text']
        
        # 2. Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # --- SIMULATED "TRADITIONAL" AUDIT ---
        
        # Check 1.1.1 (Missing Alt)
        images = soup.find_all('img')
        img_failures = 0
        for img in images:
            if not img.get('alt'):
                img_failures += 1
                
        # Check 1.4.3 (Contrast) -> Blind
        
        # Check 2.4.4 (Empty Links)
        links = soup.find_all('a')
        empty_links = 0
        for link in links:
            if not link.get_text(strip=True) and not link.find('img'):
                empty_links += 1

        results.append({
            "image_id": i,
            "method": "code_baseline",
            "1.1.1_fail_count": img_failures,
            "1.1.1_status": "FAIL" if img_failures > 0 else "PASS",
            "1.4.3_fail_count": 0,
            "1.4.3_status": "PASS",
            "2.4.4_fail_count": empty_links,
            "2.4.4_status": "FAIL" if empty_links > 0 else "PASS"
        })
        
        successful_samples += 1
        
        if successful_samples % 100 == 0:
            print(f"Audited {successful_samples} HTML inputs...")

    except StopIteration:
        print("End of dataset reached early.")
        break
    except Exception as e:
        print(f"Warning: Error processing sample {successful_samples}, retrying... {e}")
        time.sleep(1)
        continue

# Save
df = pd.DataFrame(results)
df.to_csv(OUTPUT_PATH, index=False)
print(f"✓ Code-based audit complete. Saved {len(df)} rows to {OUTPUT_PATH}")
