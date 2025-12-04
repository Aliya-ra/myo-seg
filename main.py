import config
from patient import PatientCase
import pandas as pd
import traceback
from tqdm import tqdm

def main():

    # Check if the main data folder exists
    if not config.ROOT_DIR.exists():
        print(f"Error: The directory '{config.ROOT_DIR}' was not found.")
        print("Please check config.py and ensure the path is correct.")
        return
    

    # Find all subdirectories (one per patient)
    patient_dirs = sorted([d for d in config.ROOT_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(patient_dirs)} patient directories.")
    print("=" * 60)

    all_stats = []



    for p_dir in tqdm(patient_dirs, desc="Overall Progress", unit="patient"):
        case_id = p_dir.name
        print(f"Processing {case_id}...")

        try:

            patient = PatientCase(p_dir)
            patient.load_data()

            stats = patient.run_analysis()

            if stats:
                all_stats.append(stats)
                print(f"Success. Processed {len(patient.available_arteries)} arteries.")

        except Exception as e:
            print(f"Failed:{e}")
            traceback.print_exc()
            continue
    
    print("=" * 60)



    # Reporting
    if not all_stats:
        print("No patients were successfully processed.")
        return
    summary_df = pd.DataFrame(all_stats)
    
    # Organize columns nicely
    cols = ["case_id", "total_volume_ml"]
    for art in config.POSSIBLE_ARTERIES:
        col_name = f"{art}_percent"
        if col_name in summary_df.columns:
            cols.append(col_name)
    
    # Save to CSV
    output_path = config.ROOT_DIR / "final_territory_summary_oop.csv"
    summary_df[cols].to_csv(output_path, index=False, float_format="%.2f")
    
    print(f"\nAnalysis Complete.")
    print(f"Summary saved to: {output_path}")
    
    # Print a preview
    print("\nPreview:")
    print(summary_df[cols].head().to_string(index=False))

if __name__ == "__main__":
    main() 

