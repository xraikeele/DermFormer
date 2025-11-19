
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
