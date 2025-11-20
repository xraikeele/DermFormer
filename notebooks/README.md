# DermFormer Interactive Demos

This directory contains interactive demonstrations of the paper DermFormer: Nested Multi-modal Vision Transformers for Robust Skin Cancer Detection

## Notebooks

- **DermFormer_Interactive_Demo.ipynb**: Interactive inference demo with model predictions and visualizations
- **DermFormer_Robustness_Demo.ipynb**: Robustness testing demo showing corruption and noise resilience

## Helper Scripts

- **extract_demo_from_csv.py**: Extract sample cases from Derm7pt metadata
- **sample_data/**: Directory containing demo images and metadata

## Quick Start

### 1. Activate Environment

```bash
conda activate dermformer
cd notebooks
```

### 2. Run a Notebook

**Interactive Inference Demo:**
```bash
jupyter notebook DermFormer_Interactive_Demo.ipynb
```

**Robustness Testing Demo:**
```bash
jupyter notebook DermFormer_Robustness_Demo.ipynb
```

**VS Code:**
- Open the `.ipynb` file directly in VS Code
- Select the `dermformer` kernel
- Run cells sequentially

### 3. Google Colab Deployment

To share on Google Colab:

1. Upload the notebook to your Google Drive
2. Upload the model checkpoint: `best_model_97.pth` (211MB)
3. Create a `sample_data/` folder with a few example images

**Colab modifications needed:**
- Uncomment installation cells (Section 1)
- Update model path to Google Drive location (Section 2)
- Uncomment Drive mounting code

## What the Demos Show

### Demo 1: Interactive Inference

**Technical Skills Demonstrated:**
- Multi-modal deep learning architecture
- Medical image classification
- Ensemble learning with uncertainty quantification
- Interactive visualization and interpretation
- Production ML deployment

**Features:**
1. **Multi-modal inference**: Process clinical + dermoscopic images + metadata
2. **Branch analysis**: Compare individual model branches (Clinical, Dermoscopic, Combined, Meta-Combined)
3. **Entropy weighting**: Visualize how prediction confidence affects ensemble weights
4. **Multi-task predictions**: Diagnose 8 different lesion characteristics simultaneously
5. **Case studies**: Real Derm7pt examples with ground truth comparison

### Demo 2: Robustness Testing

**Technical Skills Demonstrated:**
- Model robustness evaluation
- Systematic corruption and noise testing
- Performance degradation analysis
- Statistical visualization
- Comparative model benchmarking

**Features:**
1. **Corruption testing**: 18 distortion types (noise, blur, weather, digital artifacts)
2. **Noise sensitivity**: Gaussian noise injection at multiple intensity levels
3. **Modality-specific testing**: Separate corruption of clinical vs dermoscopic images
4. **Performance curves**: Accuracy degradation across severity levels
5. **Model comparison**: DermFormer vs baseline architectures (TFormer, NEST-DER, NEST-CLI, NEST-MMC)

### Model Performance
- Diagnosis Accuracy: **0.779**
- F-score: **0.684**
- Validation Accuracy: **0.741**
- Model Size: 211 MB (52M parameters)

## Customization

### Using Your Own Images

Update Section 7 with paths to your images:

```python
SAMPLE_CLINICAL_IMG = 'path/to/your/clinical_image.jpg'
SAMPLE_DERMOSCOPIC_IMG = 'path/to/your/dermoscopic_image.jpg'

metadata = {
    'elevation': 'raised',      # 'flat', 'raised', 'unknown'
    'sex': 'female',            # 'male', 'female', 'unknown'
    'location': 'back',         # See METADATA_ENCODINGS for options
    'lesion_type': 'atypical',  # 'typical', 'atypical'
    'age_group': '>50'          # '<30', '30-50', '>50'
}
```

### Adding More Visualizations

The notebook includes functions for:
- `visualize_prediction()`: Single task detailed view
- `visualize_all_tasks()`: All 8 tasks in grid
- `analyze_branch_contributions()`: Branch-level analysis
- `visualize_entropy_weighting()`: Entropy weighting explanation

You can create custom visualizations by accessing the `results` dictionary.

```

### Key Selling Points

Highlight in your portfolio:
- **0.779 diagnosis accuracy** on Derm7pt benchmark
- **Multi-modal fusion** of clinical, dermoscopic, and patient metadata
- **Interpretable AI** with entropy-weighted ensemble mechanism
- **Robust to real-world variations**: Maintains performance under corruption and noise
- **Multi-task learning**: 8 simultaneous classification tasks
- **Production-ready** with optimized inference pipeline

## Troubleshooting

**Model not loading:**
- Check path in Section 2: `MODEL_PATH`
- Ensure you're using the correct checkpoint: `best_model_97.pth`

**Import errors:**
- Ensure you're in the DermFormer directory
- Activate the conda environment: `conda activate dermformer`
- Check that `sys.path.insert(0, '..')` points to correct location

**CUDA out of memory:**
- Reduce batch size (model processes one image at a time in inference)
- Use CPU: `DEVICE = torch.device('cpu')`

**Images not displaying:**
- Check image paths in Section 7
- Ensure images are RGB format
- Verify PIL can open the images

## Contact

Matthew Cockayne  
Email: m.j.cockayne@keele.ac.uk  
GitHub: [@xraikeele](https://github.com/xraikeele)

---

*Created: November 2025 | Part of PhD Research Portfolio*
