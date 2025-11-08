# Watercut & Shut-in/EDP Priority Fix

## Issues Found by Azhar

1. **Watercut tidak kebaca** - Watercut data not being loaded/applied
2. **Shut-in jadi EDP** - Pipeline showing EDP when it should be Shut-in

## Root Causes

### Issue 1: Override Priority Order Was BACKWARDS ❌

**WRONG (before fix):**
```python
pred_vec = np.where(mask_wc, 12, pred_vec)      # Watercut applied first
pred_vec = np.where(mask_shutin, 11, pred_vec)  # Shut-in applied second
pred_vec = np.where(mask_edp, 10, pred_vec)     # EDP applied LAST = highest priority ❌
```

This meant **EDP had highest priority** and would override Shut-in and Watercut!

**CORRECT (after fix):**
```python
pred_vec = np.where(mask_edp, 10, pred_vec)     # EDP applied first (lowest priority)
pred_vec = np.where(mask_wc, 12, pred_vec)      # Watercut applied second
pred_vec = np.where(mask_shutin, 11, pred_vec)  # Shut-in applied LAST = highest priority ✅
```

Now **Shut-in has highest priority** as per notebook logic!

### Issue 2: Watercut Loading Path Issues

The Watercut loader was looking in hardcoded paths that might not match your file structure.

**Fixed with flexible path resolution:**
1. User-provided `--prod-data` path (file or directory)
2. Sibling of input: `Test Web/Data Produksi/{well}.csv`
3. Project standard: `Test Web/Data Produksi/{well}.csv`
4. Fallback: `prod_data.csv` in project root

## Changes Made

### File: `src/pipeline.py`

1. **Fixed override order** (lines 1042-1051):
   - EDP → Watercut → Shut-in (correct priority)
   - Added logging to show override counts

2. **Enhanced Watercut loading** (lines 1135-1219):
   - Flexible path resolution
   - Supports multiple CSV formats (Date+WC, Date+Well+WC, wide format)
   - Better error messages showing all attempted paths

3. **Added debug logging**:
   - Shows when Watercut data is loaded and how many rows
   - Shows override counts for EDP, Watercut, and Shut-in
   - Warns if Watercut data is not loaded

## Expected Results After Fix

### SKW-02
- **Before**: 19 rows showing "EDP" + 3 rows showing "EDP"
- **After**: 19 rows showing "Shut-in" + 3 rows showing "100% Watercut" ✅

### SKW-07
- **Before**: 3 rows showing "Shut-in"
- **After**: 3 rows showing "100% Watercut" ✅

### SKW-33
- **Before**: 26 rows showing "EDP" + 18 rows showing "Shut-in"
- **After**: 26 rows showing "Shut-in" + 18 rows showing "100% Watercut" ✅

### SKW-30
- **Before**: 1 row showing "EDP"
- **After**: 1 row showing "Shut-in" ✅

## How to Test

### Option 1: Run test script
```bash
prod_env/bin/python3 test_watercut_fix.py
```

### Option 2: Run individual well with INFO logging
```bash
prod_env/bin/python3 main.py \
  --well-name "SKW-02" \
  --input-file "Test Web/Data Sensor/SKW-02.csv" \
  --output-dir "test_output/SKW-02" \
  --log-level INFO
```

Look for these log messages:
- ✅ `Loaded Watercut data with X rows from ...`
- ✅ `Watercut=100% detected for X windows`
- ✅ `Override counts - EDP: X, Watercut=100%: Y, Shut-in: Z`

### Option 3: Compare outputs
```bash
prod_env/bin/python3 compare_outputs.py
```

## Priority Order Summary

| Priority | Status | Applied When |
|----------|--------|--------------|
| **1 (Highest)** | **Shut-in** | Amps=0, Freq=0, and (VR/DP/Vibration=0 OR has variation) |
| 2 | 100% Watercut | Daily WC=100% from production data |
| 3 (Lowest) | EDP | Amps=0, Freq=0, all sensors=0, no variation |

**Key Rule**: If a window matches multiple conditions, **Shut-in always wins**.

## Files Modified

- ✅ `src/pipeline.py` - Fixed override order and Watercut loading
- ✅ `main.py` - Wire `--prod-data` to pipeline
- ✅ `test_watercut_fix.py` - Test script (new)
- ✅ `WATERCUT_FIX_SUMMARY.md` - This document (new)
