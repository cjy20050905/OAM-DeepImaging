# Data Requirements for Paper Submission

## 📊 Required Data and Visualizations

### 1. Training Results (Already Have ✅)

From V8 training on server:
- ✅ Best PSNR: 30.46 dB
- ✅ Training history (losses, PSNR curves)
- ✅ Model checkpoint

**Action**: Copy from server `outputs_v8/` to local

### 2. Visualization Figures (Need to Generate 🔴)

Run on server:
```bash
cd scripts
python visualize.py --model_path ../outputs_v8/best_model_v8.pth --output_dir ../figures
```

**Expected outputs**:
- `reconstruction_results.png/pdf` - 8 sample reconstructions with error maps
- `training_curves.png/pdf` - Loss and PSNR curves
- `psnr_distribution.png/pdf` - PSNR histogram

### 3. Evaluation Metrics (Need to Generate 🔴)

Run on server:
```bash
cd scripts
python evaluate.py --model_path ../outputs_v8/best_model_v8.pth
```

**Expected outputs**:
- Mean PSNR ± std
- Median PSNR
- PSNR range [min, max]
- MAE, RMSE metrics

### 4. Ablation Study Results (Need to Run 🔴)

**Required experiments**:

#### Experiment 1: No Data Augmentation
```python
# Modify dataset.py: augment=False
# Expected PSNR: 19-20 dB
```

#### Experiment 2: No Combined Loss (L2 only)
```python
# Modify train.py: criterion = nn.MSELoss()
# Expected PSNR: 19-20 dB
```

#### Experiment 3: No Data Aug + No Combined Loss
```python
# Both modifications
# Expected PSNR: 19-20 dB (baseline)
```

**Time estimate**: 3 × 3 hours = 9 hours total

### 5. Comparison with Baseline Methods (Need to Implement 🔴)

**Required baselines**:

#### Traditional Methods:
1. **OMP (Orthogonal Matching Pursuit)**
   - Implementation: Use scikit-learn or custom
   - Expected PSNR: 12-15 dB

2. **ISTA (Iterative Shrinkage-Thresholding)**
   - Implementation: Custom (simple)
   - Expected PSNR: 15-18 dB

#### Deep Learning Methods:
3. **U-Net**
   - Implementation: Standard U-Net architecture
   - Expected PSNR: 20-25 dB

4. **LISTA-Net** (Optional)
   - Implementation: Learned ISTA
   - Expected PSNR: 22-26 dB

**Time estimate**: 2-3 days

### 6. Robustness Experiments (Optional but Recommended 🟡)

#### Different Noise Levels:
```python
noise_std = [0.0001, 0.001, 0.01, 0.1]
# Test model on each noise level
```

#### Different Number of Targets:
```python
num_targets = [1, 2, 3, 4, 5]
# Test model on each configuration
```

**Time estimate**: 1 day

### 7. Different Compression Ratios (Optional 🟡)

Test on different image sizes:
```python
image_sizes = [20, 24, 28, 32]
# Train model for each size
```

**Time estimate**: 2-3 days

## 📋 Priority Checklist

### High Priority (Must Have for ICASSP)

- [x] V8 training results (30.46 dB) ✅
- [ ] Visualization figures (reconstruction, curves, distribution) 🔴
- [ ] Evaluation metrics (mean, std, range) 🔴
- [ ] Ablation study (3 experiments) 🔴
- [ ] Baseline comparison (OMP, ISTA) 🔴

**Estimated time**: 3-4 days

### Medium Priority (Strongly Recommended)

- [ ] U-Net baseline comparison 🟡
- [ ] Robustness experiments (noise, targets) 🟡

**Estimated time**: 2-3 days

### Low Priority (Nice to Have)

- [ ] LISTA-Net comparison 🟢
- [ ] Different compression ratios 🟢
- [ ] Real data validation (if available) 🟢

**Estimated time**: 3-5 days

## 🚀 Immediate Action Plan

### Step 1: Generate Visualizations (30 minutes)

```bash
# On server
cd /root/OAM/official_release/scripts
python visualize.py
```

### Step 2: Run Evaluation (30 minutes)

```bash
# On server
cd /root/OAM/official_release/scripts
python evaluate.py
```

### Step 3: Ablation Study (9 hours)

Train 3 ablation models:
1. No augmentation
2. No combined loss
3. Baseline (no aug + no combined loss)

### Step 4: Implement Baselines (2 days)

Implement and test:
1. OMP
2. ISTA
3. U-Net

## 📊 Expected Results Summary

| Experiment | PSNR (Expected) | Status |
|------------|-----------------|--------|
| **Ours (Full)** | **30.46 dB** | ✅ Done |
| Ours (No Aug) | 19-20 dB | 🔴 Need |
| Ours (No Loss) | 19-20 dB | 🔴 Need |
| Baseline | 19-20 dB | 🔴 Need |
| OMP | 12-15 dB | 🔴 Need |
| ISTA | 15-18 dB | 🔴 Need |
| U-Net | 20-25 dB | 🟡 Optional |

## 📝 Data Format for Paper

### Table 1: Main Results

```latex
\begin{table}
\caption{Comparison with State-of-the-Art Methods}
\begin{tabular}{lccc}
\hline
Method & Type & PSNR (dB) & Parameters \\
\hline
OMP & Traditional & 13.2 & - \\
ISTA & Optimization & 16.5 & - \\
U-Net & Deep Learning & 22.3 & 7.0M \\
\textbf{Ours} & \textbf{End-to-End} & \textbf{30.46} & \textbf{0.5M} \\
\hline
\end{tabular}
\end{table}
```

### Table 2: Ablation Study

```latex
\begin{table}
\caption{Ablation Study Results}
\begin{tabular}{ccc}
\hline
Data Aug & Combined Loss & PSNR (dB) \\
\hline
\xmark & \xmark & 19.35 \\
\cmark & \xmark & 28-29 \\
\xmark & \cmark & 20-21 \\
\cmark & \cmark & \textbf{30.46} \\
\hline
\end{tabular}
\end{table}
```

## 🎯 Timeline

- **Week 1**: Visualizations + Evaluation + Ablation (High Priority)
- **Week 2**: Baseline implementations (OMP, ISTA, U-Net)
- **Week 3**: Robustness experiments + Paper writing
- **Week 4**: Paper revision + Submission

**Target submission**: ICASSP 2027 (October 2026)

---

**Last Updated**: 2026-03-06
