import os
import numpy as np
import torch
import wfdb
from src.dataset import DS1_RECORDS, DS2_RECORDS, AAMI_MAPPING, CLASS_NAMES, load_mitdb_beats, get_dataloaders

def execute_phase1(data_dir="./data/mitdb"):
    print("==========================================================================")
    print("PHASE 1: Dataset Ingestion & Inter-Patient Preprocessing (MIT-BIH PhysioNet)")
    print("==========================================================================")
    
    os.makedirs(data_dir, exist_ok=True)
    all_records = DS1_RECORDS + DS2_RECORDS
    records_str = [str(r) for r in all_records]
    
    # 1. PhysioNet Database Download
    missing = [r for r in records_str if not os.path.exists(os.path.join(data_dir, f"{r}.dat"))]
    if missing:
        print(f"Downloading {len(missing)} missing PhysioNet MIT-BIH records to {data_dir}...")
        try:
            wfdb.dl_database('mitdb', dl_dir=data_dir, records=records_str)
            print("PhysioNet download successfully completed.")
        except Exception as e:
            print(f"Download exception: {e}")
    else:
        print("All 44 PhysioNet MIT-BIH clinical records are present locally.")
        
    # 2. Inter-Patient Data Splitting
    print("\n--- Verifying Patient Record Independence ---")
    ds1_set = set(DS1_RECORDS)
    ds2_set = set(DS2_RECORDS)
    overlap = ds1_set.intersection(ds2_set)
    print(f"DS1 (Training Records): {len(DS1_RECORDS)} records")
    print(f"DS2 (Testing Records) : {len(DS2_RECORDS)} records")
    print(f"Record Overlap Count  : {len(overlap)} (Zero Data Leakage Verified: {len(overlap) == 0})")
    
    # 3. Extract Beats & Class Frequencies
    print("\n--- Extracting Segmented Beats (Lead II, 250 Hz, L=256) ---")
    ds1_signals, ds1_labels = load_mitdb_beats(data_dir=data_dir, records=DS1_RECORDS)
    ds2_signals, ds2_labels = load_mitdb_beats(data_dir=data_dir, records=DS2_RECORDS)
    
    print(f"\nDS1 (Training Set) Total Beats : {len(ds1_labels)} | Signal Shape: {ds1_signals.shape}")
    print(f"DS2 (Testing Set)  Total Beats : {len(ds2_labels)} | Signal Shape: {ds2_signals.shape}")
    
    # 4. Compute Class Frequencies & Weights
    print("\n--- Computing Class Distributions & Inverse Weights (DS1) ---")
    counts_ds1 = np.bincount(ds1_labels, minlength=5)
    counts_ds2 = np.bincount(ds2_labels, minlength=5)
    
    total_ds1 = len(ds1_labels)
    inverse_weights = total_ds1 / (5.0 * counts_ds1 + 1e-6)
    
    print("Class Distribution Breakdown:")
    print(f"{'Class':<8}{'Name':<25}{'DS1 Count':<12}{'DS1 %':<10}{'DS2 Count':<12}{'DS1 Loss Weight':<15}")
    print("-" * 80)
    for c in range(5):
        c_name = CLASS_NAMES[c]
        c_desc = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Paced/Unknown (Q)'][c]
        pct = (counts_ds1[c] / total_ds1) * 100.0
        print(f"{c:<8}{c_desc:<25}{counts_ds1[c]:<12}{pct:<10.2f}%{counts_ds2[c]:<12}{inverse_weights[c]:<15.4f}")
        
    # 5. Build & Test PyTorch DataLoaders
    print("\n--- Building & Validating PyTorch DataLoaders ---")
    train_loader, val_loader, test_loader = get_dataloaders(data_dir=data_dir, batch_size=64, use_interpatient=True)
    sample_x, sample_y = next(iter(train_loader))
    print(f"Train Batch Shape : x={sample_x.shape}, y={sample_y.shape}")
    print(f"Validation Batches: {len(val_loader)} batches")
    print(f"Testing Batches   : {len(test_loader)} batches")
    
    print("\n==========================================================================")
    print("PHASE 1 COMPLETED SUCCESSFULLY: Dataset ready for model training.")
    print("==========================================================================")
    
    return {
        'ds1_count': len(ds1_labels),
        'ds2_count': len(ds2_labels),
        'counts_ds1': counts_ds1.tolist(),
        'counts_ds2': counts_ds2.tolist(),
        'inverse_weights': inverse_weights.tolist()
    }

if __name__ == "__main__":
    execute_phase1()
