"""
Extract sample entries from WebSight v0.2 dataset for use in RDF examples.
"""
from datasets import load_dataset
import hashlib
import json

def get_sha256(content):
    """Compute SHA-256 hash of string content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def extract_samples():
    """Load WebSight and extract 2 representative samples."""
    print("Loading WebSight v0.2 dataset...")
    ds = load_dataset("HuggingFaceM4/WebSight", "v0.2", split="train", streaming=True)
    
    samples = []
    for idx, item in enumerate(ds):
        if idx >= 2:  # Only extract 2 samples
            break
        
        html = item.get('text', '')
        img = item.get('image', None)
        
        # Compute hash of HTML content
        html_hash = get_sha256(html)
        
        # Extract some basic info
        sample_info = {
            'index': idx,
            'html_snippet': html[:500] if html else '',
            'html_length': len(html),
            'html_sha256': html_hash,
            'has_image': img is not None
        }
        
        samples.append(sample_info)
        print(f"\nSample {idx}:")
        print(f"  HTML length: {sample_info['html_length']} chars")
        print(f"  SHA-256: {html_hash}")
        print(f"  Snippet: {html[:200]}...")
    
    # Save to JSON
    with open('websight_samples.json', 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print("\n\nSamples saved to websight_samples.json")
    return samples

if __name__ == '__main__':
    samples = extract_samples()
