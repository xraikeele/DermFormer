# DermFormer Interactive Demo

This directory contains interactive demonstrations of the DermFormer model for your portfolio.

## Files

- **DermFormer_Interactive_Demo.ipynb**: Main interactive notebook with full functionality
- **setup_demo_data.py**: Helper script to prepare sample cases for demonstration

## Quick Start

### 1. Prepare Sample Data

```bash
cd notebooks
python setup_demo_data.py
```

This will copy a few representative cases from the Derm7pt dataset for demonstration.

### 2. Run the Notebook

**Local Jupyter:**
```bash
jupyter notebook DermFormer_Interactive_Demo.ipynb
```

**JupyterLab:**
```bash
jupyter lab DermFormer_Interactive_Demo.ipynb
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
4. Share the Colab link in your portfolio

**Colab modifications needed:**
- Uncomment installation cells (Section 1)
- Update model path to Google Drive location (Section 2)
- Uncomment Drive mounting code

## What the Demo Shows

### Technical Skills
- ✅ PyTorch model implementation
- ✅ Multi-modal deep learning
- ✅ Medical image analysis
- ✅ Ensemble learning & uncertainty quantification
- ✅ Data visualization & interpretation
- ✅ Production ML deployment

### Interactive Features
1. **Multi-modal inference**: Process clinical + dermoscopic images + metadata
2. **Branch analysis**: Compare individual model branches (Clinical, Dermoscopic, Combined, Meta)
3. **Entropy weighting**: Visualize how uncertainty affects ensemble predictions
4. **Multi-task predictions**: Diagnose 8 different lesion characteristics
5. **Beautiful visualizations**: Professional matplotlib/seaborn plots

### Model Performance
- Test Accuracy: **76.68%**
- Best Epoch: 97 (early stopping at 147)
- Model Size: 211 MB
- Parameters: ~52M trainable

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

## Portfolio Integration

### Website Embedding

Add to your portfolio website:

```html
<h3>DermFormer: Multi-Modal Skin Lesion Diagnosis</h3>
<p>Interactive deep learning demo for medical image analysis</p>
<a href="link-to-colab-notebook" class="btn">🚀 Try Live Demo</a>
<a href="link-to-github" class="btn">💻 View Code</a>
```

### Key Selling Points

Highlight in your portfolio:
- **76.68% accuracy** on dermatology benchmark
- **Multi-modal fusion** of clinical, dermoscopic, and metadata
- **Interpretable AI** with entropy-weighted ensemble
- **Production-ready** with optimized inference
- **8 simultaneous tasks** (diagnosis + 7 lesion characteristics)

## Next Steps

1. ✅ **Run the notebook locally** to verify everything works
2. ✅ **Test with sample cases** from Derm7pt dataset
3. ✅ **Deploy to Google Colab** for public access
4. ✅ **Add to your portfolio website** with live demo link
5. 🔄 **Create variations**:
   - Robustness testing notebook (Portfolio Idea #2)
   - Attention visualization notebook (Portfolio Idea #3)
   - Tutorial notebook (Portfolio Idea #9)

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

Questions or issues? Contact Matthew Cockayne at [your-email]

---

*Created: November 2025 | Part of PhD Research Portfolio*
