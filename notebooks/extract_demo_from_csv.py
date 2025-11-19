#!/usr/bin/env python3
"""
Extract Demo Data from Derm7pt CSV Metadata

This script reads the Derm7pt metadata CSV and extracts properly paired
clinical + dermoscopic images with complete metadata for the demo notebook.

Usage:
    python3 extract_demo_from_csv.py <path_to_csv>

Author: Matthew Cockayne
Date: November 2025
"""

import pandas as pd
import shutil
import json
from pathlib import Path
import sys

# Paths
DERM7PT_IMAGES_PATH = Path("/home/matthewcockayne/datasets/Derm7pt/release_v0/release_v0/images")
DEMO_DATA_PATH = Path("sample_data")

# Metadata mappings (for encoding)
ELEVATION_MAP = {'flat': 'flat', 'palpable': 'raised', 'nodular': 'raised'}
SEX_MAP = {'male': 'male', 'female': 'female'}
LOCATION_MAP = {
    'abdomen': 'abdomen',
    'upper limbs': 'upper extremity',
    'lower limbs': 'lower extremity',
    'chest': 'chest',
    'back': 'back',
    'face': 'face',
    'neck': 'neck',
    'scalp': 'scalp'
}

DIAGNOSIS_MAP = {
    'basal cell carcinoma': 'Basal Cell Carcinoma',
    'blue nevus': 'Blue Nevus',
    'clark nevus': 'Clark Nevus',
    'dermatofibroma': 'Dermatofibroma',
    'melanoma': 'Melanoma',
    'combined nevus': 'Clark Nevus',  # Treat as Clark Nevus
    'congenital nevus': 'Clark Nevus',
    'melanosis': 'Clark Nevus',
    'miscellaneous': 'Dermatofibroma'
}

def get_age_group(elevation_level):
    """Infer age group from elevation (simplified heuristic)."""
    if elevation_level in ['low', 'medium']:
        return '30-50'
    else:
        return '>50'

def find_image_path(image_filename):
    """Find the full path to an image file."""
    if not image_filename or pd.isna(image_filename):
        return None
    
    # Extract folder name (e.g., "NEL" from "NEL/NEL02")
    parts = image_filename.split('/')
    if len(parts) < 2:
        return None
    
    folder = parts[0]
    filename = parts[1]
    
    full_path = DERM7PT_IMAGES_PATH / folder / filename
    
    if full_path.exists():
        return full_path
    else:
        print(f"  ⚠️  Image not found: {full_path}")
        return None

def extract_demo_data(csv_path, num_cases=10):
    """
    Extract demo cases from CSV metadata.
    
    Args:
        csv_path: Path to the metadata CSV file
        num_cases: Number of cases to extract (default: 10)
    """
    print("="*70)
    print("DermFormer Demo Data Extraction from CSV")
    print("="*70)
    
    # Read CSV
    print(f"\n📄 Reading CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} cases from CSV")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False
    
    # Display available columns
    print(f"\n📋 Available columns: {list(df.columns)}")
    
    # Create demo directory
    DEMO_DATA_PATH.mkdir(exist_ok=True)
    print(f"✅ Created demo directory: {DEMO_DATA_PATH}")
    
    # Filter for cases with both clinical and dermoscopic images
    df_filtered = df[
        df['clinic'].notna() & 
        df['derm'].notna() &
        df['diagnosis'].notna()
    ].copy()
    
    print(f"\n🔍 Found {len(df_filtered)} cases with paired images")
    
    # Try to get diverse diagnosis types
    diagnosis_counts = df_filtered['diagnosis'].value_counts()
    print(f"\n📊 Diagnosis distribution:")
    for diag, count in diagnosis_counts.items():
        print(f"   {diag}: {count} cases")
    
    # Sample diverse cases (2 per diagnosis type if possible)
    sampled_cases = []
    for diagnosis in df_filtered['diagnosis'].unique():
        diag_cases = df_filtered[df_filtered['diagnosis'] == diagnosis].head(2)
        sampled_cases.append(diag_cases)
    
    df_demo = pd.concat(sampled_cases).head(num_cases)
    
    print(f"\n📦 Extracting {len(df_demo)} demo cases...")
    
    # Extract cases
    metadata = []
    success_count = 0
    
    for idx, row in df_demo.iterrows():
        case_num = row['case_num']
        diagnosis_raw = row['diagnosis']
        
        print(f"\n  📸 Case {case_num}: {diagnosis_raw}")
        
        # Find image paths
        clinical_src = find_image_path(row['clinic'])
        dermo_src = find_image_path(row['derm'])
        
        if not clinical_src or not dermo_src:
            print(f"     ✗ Missing images, skipping")
            continue
        
        # Copy images
        case_id = f"case_{case_num:03d}"
        clinical_dst = DEMO_DATA_PATH / f"{case_id}_clinical.jpg"
        dermo_dst = DEMO_DATA_PATH / f"{case_id}_dermoscopic.jpg"
        
        try:
            shutil.copy2(clinical_src, clinical_dst)
            shutil.copy2(dermo_src, dermo_dst)
            print(f"     ✓ Clinical: {clinical_src.name}")
            print(f"     ✓ Dermoscopic: {dermo_src.name}")
        except Exception as e:
            print(f"     ✗ Copy error: {e}")
            continue
        
        # Parse metadata
        elevation = ELEVATION_MAP.get(str(row.get('elevation', 'unknown')).lower(), 'unknown')
        sex = SEX_MAP.get(str(row.get('sex', 'unknown')).lower(), 'unknown')
        location = LOCATION_MAP.get(str(row.get('location', 'unknown')).lower(), 'unknown')
        diagnosis = DIAGNOSIS_MAP.get(str(diagnosis_raw).lower(), diagnosis_raw)
        
        # Determine lesion type from pigment network
        pigment_net = str(row.get('pigment_network', 'absent')).lower()
        lesion_type = 'atypical' if pigment_net in ['atypical', 'irregular'] else 'typical'
        
        # Age group heuristic
        elevation_level = str(row.get('level_of_diagnostic_difficulty', 'medium')).lower()
        age_group = get_age_group(elevation_level)
        
        # Store metadata
        case_metadata = {
            "case_id": case_id,
            "case_num": int(case_num),
            "diagnosis": diagnosis,
            "clinical_img": clinical_dst.name,
            "dermoscopic_img": dermo_dst.name,
            "metadata": {
                "elevation": elevation,
                "sex": sex,
                "location": location,
                "lesion_type": lesion_type,
                "age_group": age_group
            },
            "clinical_features": {
                "seven_point": str(row.get('seven_point', 'unknown')),
                "pigment_network": str(row.get('pigment_network', 'unknown')),
                "streaks": str(row.get('streaks', 'unknown')),
                "pigmentation": str(row.get('pigmentation', 'unknown')),
                "regression": str(row.get('regression_structures', 'unknown')),
                "dots_and_globules": str(row.get('dots_and_globules', 'unknown')),
                "blue_whitish_veil": str(row.get('blue_whitish_veil', 'unknown')),
                "vascular_structures": str(row.get('vascular_structures', 'unknown'))
            },
            "original_paths": {
                "clinical": str(row['clinic']),
                "dermoscopic": str(row['derm'])
            }
        }
        
        metadata.append(case_metadata)
        success_count += 1
        print(f"     ✓ Metadata: {sex}, {age_group}, {location}, {elevation}")
    
    # Save metadata JSON
    metadata_file = DEMO_DATA_PATH / "cases_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, indent=2, fp=f)
    
    print(f"\n{'='*70}")
    print(f"✅ Successfully extracted {success_count} demo cases")
    print(f"✅ Metadata saved to: {metadata_file}")
    print(f"{'='*70}")
    
    # Create summary
    print("\n📊 Demo Dataset Summary:")
    print(f"   Total cases: {success_count}")
    
    if metadata:
        diag_dist = {}
        for case in metadata:
            diag = case['diagnosis']
            diag_dist[diag] = diag_dist.get(diag, 0) + 1
        
        print("\n   Diagnosis distribution:")
        for diag, count in sorted(diag_dist.items()):
            print(f"      {diag}: {count}")
        
        print("\n   Metadata coverage:")
        sex_dist = {}
        loc_dist = {}
        for case in metadata:
            sex_dist[case['metadata']['sex']] = sex_dist.get(case['metadata']['sex'], 0) + 1
            loc_dist[case['metadata']['location']] = loc_dist.get(case['metadata']['location'], 0) + 1
        
        print(f"      Sex: {dict(sex_dist)}")
        print(f"      Locations: {dict(loc_dist)}")
    
    # Create loading example
    example_code = '''
# Example: Load and use demo data in notebook

import json

# Load all demo cases
with open('notebooks/sample_data/cases_metadata.json', 'r') as f:
    demo_cases = json.load(f)

print(f"Loaded {len(demo_cases)} demo cases")

# Select a case (e.g., first melanoma case)
case = [c for c in demo_cases if c['diagnosis'] == 'Melanoma'][0]

# Or select by index
case = demo_cases[0]

# Extract paths and metadata for inference
SAMPLE_CLINICAL_IMG = f'notebooks/sample_data/{case["clinical_img"]}'
SAMPLE_DERMOSCOPIC_IMG = f'notebooks/sample_data/{case["dermoscopic_img"]}'
metadata = case['metadata']

print(f"Case {case['case_num']}: {case['diagnosis']}")
print(f"Clinical: {SAMPLE_CLINICAL_IMG}")
print(f"Dermoscopic: {SAMPLE_DERMOSCOPIC_IMG}")
print(f"Metadata: {metadata}")

# Run inference
results = predict(SAMPLE_CLINICAL_IMG, SAMPLE_DERMOSCOPIC_IMG, metadata)
visualize_prediction(results, task='diag')
'''
    
    example_file = DEMO_DATA_PATH / "loading_example.py"
    with open(example_file, 'w') as f:
        f.write(example_code)
    
    print(f"\n✅ Loading example saved to: {example_file}")
    
    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    print("1. Open DermFormer_Interactive_Demo.ipynb")
    print("2. In Section 7, use this code to load cases:")
    print("   ")
    print("   import json")
    print("   with open('sample_data/cases_metadata.json') as f:")
    print("       cases = json.load(f)")
    print("   case = cases[0]  # Select any case")
    print("   ")
    print("3. Run inference on multiple cases in a loop")
    print("4. Compare predictions across different diagnoses")
    print("="*70 + "\n")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 extract_demo_from_csv.py <path_to_metadata.csv>")
        print("\nExample:")
        print("  python3 extract_demo_from_csv.py /path/to/derm7pt_metadata.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not Path(csv_path).exists():
        print(f"❌ Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    success = extract_demo_data(csv_path, num_cases=10)
    
    if not success:
        print("\n❌ Extraction failed. Please check errors above.")
        sys.exit(1)
