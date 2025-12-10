# Targeted Improvements for DO 3 & DO 4 Performance

## Overview
This document summarizes the minimal, targeted changes made to improve the dice score performance of LoRA training sites (DO 3 & DO 4) without requiring significant codebase restructuring.

## Changes Made

### 1. Enhanced Point Prompt Generation (`task.py` - `_fit_lora` method)
**What changed:**
- **Before**: Used only centroid point from mask
- **After**: Uses centroid + random foreground point (2 points per image when mask is large enough)

**Why this helps:**
- Provides richer training signal to SAM2
- Increases prompt diversity for better generalization
- Helps model learn from multiple perspectives of the same mask

**Implementation details:**
- If mask has >1 foreground pixel, uses centroid + random point
- Falls back to single centroid if mask is tiny
- Handles variable-length point arrays by padding to max length in batch

### 2. Improved Loss Function (`task.py` - `_fit_lora` method)
**What changed:**
- **Before**: `loss = dice_loss + bce` (equal weighting)
- **After**: `loss = 0.5 * dice + 0.3 * bce + 0.2 * focal`

**New addition:**
- Added `focal_loss()` function to handle class imbalance better
- Focal loss focuses learning on hard examples

**Why this helps:**
- Dice loss: Good for segmentation metrics
- BCE: Provides stable gradients
- Focal loss: Handles class imbalance and hard examples (important for medical imaging where foreground can be sparse)

### 3. Learning Rate Scheduling (`task.py` - `_fit_lora` method)
**What changed:**
- **Before**: Fixed learning rate throughout training
- **After**: Cosine annealing scheduler that gradually reduces LR

**Implementation:**
- `CosineAnnealingLR` with `T_max=local_epochs` and `eta_min=learning_rate * 0.1`
- LR decreases from initial value to 10% of initial by end of training

**Why this helps:**
- Allows more aggressive learning early, fine-tuning later
- Better convergence for LoRA adapters
- Standard practice in fine-tuning large vision models

### 4. Better Default Hyperparameters (`server_app.py`)
**What changed:**
- **Before**: `local_epochs=3`, `learning_rate=1e-4`
- **After**: `local_epochs=5`, `learning_rate=5e-5`

**Why this helps:**
- More epochs: Gives LoRA adapters more time to adapt to site-specific data
- Lower LR: More stable training, prevents overshooting (important for LoRA on vision transformers)

## Expected Impact

These changes should improve performance especially for DO 3 & DO 4 because:

1. **Better prompts** → More training signal per image → Faster learning
2. **Better loss** → Handles class imbalance better → Better on sparse foregrounds
3. **LR scheduling** → Better convergence → Higher final performance
4. **More epochs + lower LR** → More stable adaptation → Less overfitting risk

## Files Modified

1. `fl-sam2-lora/fl-sam2-lora/task.py`:
   - Added `focal_loss()` function (after `dice_loss()`)
   - Enhanced point prompt generation in `_fit_lora()`
   - Added learning rate scheduling in `_fit_lora()`
   - Improved loss calculation in `_fit_lora()`

2. `fl-sam2-lora/fl-sam2-lora/server_app.py`:
   - Updated default `local_epochs` from 3 to 5
   - Updated default `learning_rate` from 1e-4 to 5e-5

## Testing Recommendations

1. **Baseline comparison**: Run with original code to establish baseline dice scores
2. **After changes**: Run same experiment and compare:
   - DO_3 dice score (should increase from ~0.36)
   - DO_4 dice score (should increase from ~0.29)
   - Global dice score (should increase from ~0.47)
3. **Monitor training**: Check that training loss decreases smoothly and dice increases

## Next Steps (Optional Future Improvements)

If these changes don't provide sufficient improvement, consider:

1. **LoRA rank increase**: Increase from 16 to 32 for DO 3 & 4 (requires site identification)
2. **Data augmentation**: Add rotation, flipping, intensity augmentation for LoRA sites
3. **Client weighting**: Increase aggregation weight for DO 3 & 4 in FedAvg
4. **Personalization**: Post-FL local fine-tuning step for each site

## Notes

- These changes are **minimal and non-invasive** - they don't require restructuring the codebase
- All changes are **backward compatible** - existing code will still work
- Changes benefit **all LoRA sites**, but will have the most impact on underperforming sites (DO 3 & 4)
- The improvements follow best practices from the literature for SAM2 fine-tuning and LoRA training

