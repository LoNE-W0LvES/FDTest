# Kaggle Submission Guide

## Method 1: Upload Model (Recommended - Easiest)

### Step 1: Prepare Model
```bash
python prepare_for_kaggle.py
```
This creates `model_kaggle.pth` (cleaned checkpoint)

### Step 2: Upload to Kaggle
1. Go to kaggle.com → Your Profile → Datasets
2. Click "New Dataset"
3. Upload `model_kaggle.pth`
4. Name it: "my-forgery-model"
5. Make Public

### Step 3: Create Submission Notebook
1. Go to Competition page → Code → New Notebook
2. Upload `kaggle_submission.ipynb` (or copy-paste cells)
3. Add Data:
   - Click "+ Add Data"
   - Search "recodai-luc-scientific" (competition data)
   - Search "my-forgery-model" (your model)
4. Update MODEL_PATH in notebook:
   ```python
   MODEL_PATH = '/kaggle/input/my-forgery-model/model_kaggle.pth'
   ```

### Step 4: Run & Submit
1. Click "Run All"
2. Wait for completion (~5-10 minutes)
3. Click "Submit to Competition"
4. Done! 🎉

---

## Method 2: TorchScript (If Method 1 Fails)

### Step 1: Convert Model
```bash
python convert_to_torchscript.py
```
Creates `model_traced.pt`

### Step 2: Upload & Use
Same as Method 1, but use `model_traced.pt` instead

Load in notebook:
```python
model = torch.jit.load(MODEL_PATH)
model.eval()
```

---

## Files You Need

**For Local:**
- `prepare_for_kaggle.py` - Prepare model
- `convert_to_torchscript.py` - Alternative conversion

**For Kaggle:**
- `kaggle_submission.ipynb` - Submission notebook
- `model_kaggle.pth` - Your trained model

---

## Quick Checklist

- [ ] Run `prepare_for_kaggle.py` locally
- [ ] Upload `model_kaggle.pth` to Kaggle as dataset
- [ ] Create new notebook in competition
- [ ] Add competition data + your model as inputs
- [ ] Upload/paste `kaggle_submission.ipynb`
- [ ] Update MODEL_PATH
- [ ] Run All
- [ ] Submit

---

## Common Issues

**Issue: Model too large (>100MB)**
Solution: Use TorchScript (usually smaller)

**Issue: Out of memory in notebook**
Solution: Reduce BATCH_SIZE in notebook

**Issue: Can't find model file**
Solution: Check MODEL_PATH matches your dataset name

---

## Expected Output

```
Test images: 500
Generating predictions...
100%|██████████| 63/63 [02:30<00:00]
✓ Generated 500 predictions
============================================================
Submission saved to /kaggle/working/submission.csv
Total: 500
Authentic: 350
Forged: 150
============================================================
```

That's it! Super simple.
