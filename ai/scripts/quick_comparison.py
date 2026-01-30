
import pandas as pd
from pathlib import Path

# Paths
DATA_DIR = Path("data/results")
BASELINE_PATH = DATA_DIR / "code_baseline_results.csv"
VLM_PATH = DATA_DIR / "zero_shot_results.parquet"

print("--- PRELIMINARY ANALYSIS: LLaVA-7B vs CODE BASELINE ---")

try:
    # Load Data
    df_base = pd.read_csv(BASELINE_PATH)
    try:
        df_vlm = pd.read_parquet(VLM_PATH)
    except:
        print("Warning: VLM results not readable or empty yet.")
        exit()

    # 1. Total Samples
    n_base = len(df_base)
    if 'screenshot' in df_vlm.columns:
        n_vlm_images = df_vlm['screenshot'].nunique()
    elif 'image_id' in df_vlm.columns:
        n_vlm_images = df_vlm['image_id'].nunique()
    else:
        n_vlm_images = 0
    
    print(f"Baseline Samples: {n_base}")
    print(f"VLM Processed Samples: {n_vlm_images}")

    # 2. Failure Rates by Criterion
    print("\n[ Failure Detection Rates ]")
    print(f"{'Criterion':<10} | {'Baseline (Code)':<15} | {'VLM (Visual)':<15} | {'Delta':<10}")
    print("-" * 60)
    
    criteria = ["1.1.1", "1.4.3", "2.4.4"]
    
    # Pre-process VLM data: extracting status and criterion_id from 'evaluation' column
    vlm_records = []
    vlm_errors = 0
    if not df_vlm.empty and 'evaluation' in df_vlm.columns:
        for _, row in df_vlm.iterrows():
            eval_data = row['evaluation']
            # Expected structure: dict with 'page_evaluations' -> {crit_id: {error, raw_response} ...}
            if isinstance(eval_data, dict):
                page_evals = eval_data.get('page_evaluations', {})
                for crit_id, res in page_evals.items():
                    if 'error' in res:
                        vlm_errors += 1
                        # Try to recover from raw_response if possible?
                        # For now, just mark error
                    else:
                        vlm_records.append({
                            'criterion_id': crit_id,
                            'status': res.get('status')
                        })
            
    # Calculate global error rate
    total_evals_attempted = len(df_vlm) * 2 # Assuming 2 criteria per page? No, 7 criteria!
    # But num_criteria was 2 in the debug sample?
    # Let's count properly later.
    
    print(f"\nVLM Generation Errors: {vlm_errors} (Raw JSON failures)")
    
    df_vlm_processed = pd.DataFrame(vlm_records)
    
    
    df_vlm_processed = pd.DataFrame(vlm_records)
    
    if not df_vlm_processed.empty and 'status' in df_vlm_processed.columns:
        print("\nDEBUG: Unique VLM Status Values:", df_vlm_processed['status'].unique())
        # Normalize status
        df_vlm_processed['status'] = df_vlm_processed['status'].str.upper().str.strip()

    for crit in criteria:

        # Baseline Rate
        base_col = f"{crit}_status"
        if base_col in df_base.columns:
            base_fails = df_base[df_base[base_col] == "FAIL"].shape[0]
            base_rate = (base_fails / n_base) * 100
        else:
            base_rate = 0.0

        # VLM Rate
        if not df_vlm_processed.empty:
            vlm_subset = df_vlm_processed[df_vlm_processed['criterion_id'] == crit]
            vlm_total = len(vlm_subset)
            if vlm_total > 0:
                vlm_fails = vlm_subset[vlm_subset['status'] == "FAIL"].shape[0]
                vlm_rate = (vlm_fails / vlm_total) * 100
            else:
                vlm_rate = 0.0
        else:
            vlm_rate = 0.0
            
        print(f"{crit:<10} | {base_rate:>5.1f}%          | {vlm_rate:>5.1f}%          | {vlm_rate - base_rate:>+5.1f}%")

    print("-" * 60)
    print("Interpretation:")
    print(" - Positive Delta: VLM found errors the Code Parser missed (Visual Blind Spots).")
    print(" - Negative Delta: VLM missed errors (False Negatives) or Code Parser is stricter.")

except Exception as e:
    print(f"Analysis Error: {e}")
