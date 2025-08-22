# Short Run Detection in XmR Charts: Mathematical Foundations

## Overview

This document explains the mathematical basis for short run detection in XmR (Individuals and Moving Range) charts, including the statistical constants used for limit calculations and the probability theory underlying the "3 of 4 points closer to limits than center" rule.

## XmR Chart Limit Calculations

### Constants Used

XmR charts use different constants depending on whether the mean or median is chosen as the central tendency measure:

**For Mean (`xmr_function="mean"`):**
- Moving Range Upper Limit constant: **3.268**
- Natural Process Limit constant: **2.660**

**For Median (`xmr_function="median"`):**
- Moving Range Upper Limit constant: **3.865** 
- Natural Process Limit constant: **3.145**

### The 3-Sigma Foundation

XmR charts are based on **3-sigma control limits**, which capture approximately **99.73%** of normal process variation. This means:
- Only **0.27%** (about 1 in 370) of points should fall outside the limits due to random chance
- Points outside these limits likely indicate **special cause variation** requiring investigation
- This creates a balance between detecting real problems and avoiding false alarms

### Mathematical Derivation of the 2.660 Constant

**What it represents**: The 2.660 constant converts the average moving range into an estimate of 3 standard deviations.

**Step-by-step derivation**:

1. **Moving Range Relationship**: In a stable process, the average moving range (mR̄) has a known relationship to the process standard deviation:
   ```
   mR̄ ≈ 1.128 × σ
   ```
   This 1.128 factor (called d₂) comes from the expected value of the range for samples of size 2.

2. **Estimating Standard Deviation**: 
   ```
   σ = mR̄ ÷ 1.128
   ```

3. **3-Sigma Limits Calculation**:
   ```
   3σ = 3 × (mR̄ ÷ 1.128) = (3 ÷ 1.128) × mR̄ = 2.660 × mR̄
   ```

**Why this matters**: Instead of directly calculating the standard deviation (which can be influenced by trends or shifts), we use the moving range method, which is more robust for individual measurements.

### Mathematical Derivation of the 3.268 Constant

**What it represents**: The 3.268 constant sets the upper limit for the moving range chart itself.

**Derivation**:
1. The moving ranges also have their own variability
2. The standard deviation of moving ranges ≈ **0.853 × σ**
3. The upper limit becomes: **mR̄ + 3 × (0.853 × σ) = mR̄ × 3.268**

## Short Run Detection Rule: 3 of 4 Points Beyond 1.5σ

### Current Implementation

The code calculates **midrange thresholds** (halfway between center line and control limits):

```python
# Calculate midranges 
upper_midrange = [mid + ((upper - mid) / 2) for mid, upper in zip(period_mid, period_upper)]
lower_midrange = [mid - ((mid - lower) / 2) for mid, lower in zip(period_mid, period_lower)]

# Test if points are closer to limits than center
run_test_upper = [y > um for y, um in zip(period_y, upper_midrange)] 
run_test_lower = [y < lm for y, lm in zip(period_y, lower_midrange)]
```

This midpoint approach is equivalent to the **1.5σ threshold** used in Western Electric Rules.

### Why 1.5σ?

**XmR Control Limits**: 
- Upper: `mean + 2.660 × mean_mR` (approximately 3σ)
- Lower: `mean - 2.660 × mean_mR` (approximately 3σ)

**Midrange Calculation**: 
- Upper midrange: `mean + (2.660 × mean_mR) / 2 = mean + 1.330 × mean_mR`
- This equals approximately **1.5σ** from the center line

### Probability Mathematics

**Distribution Assumptions**: The calculations assume a **normal distribution**, but XmR charts are remarkably robust to deviations from normality.

**Single Point Probability**:
- Probability of a point falling beyond ±1.5σ: **~13.4%** (6.7% in each tail)

**3 of 4 Points Probability** (using binomial distribution with p = 0.067):

```
P(exactly 3 of 4) = C(4,3) × (0.067)³ × (0.933)¹ ≈ 0.0011 (0.11%)
P(exactly 4 of 4) = C(4,4) × (0.067)⁴ × (0.933)⁰ ≈ 0.00002 (0.002%)
P(3 or more of 4) ≈ 0.11%
```

### Why 3 of 4 (not 2 of 4 or 4 of 4)?

**Statistical Optimization**:
1. **2 of 4**: Too sensitive - probability ≈ 3.2%, causing excessive false alarms
2. **4 of 4**: Too restrictive - probability ≈ 0.002%, missing subtle shifts  
3. **3 of 4**: Optimal balance - probability ≈ 0.11%, sensitive but not noisy

### Type I Error Rate

The 0.11% false alarm rate means:
- Only ~1 in 900 stable sequences will trigger false alarms
- Much more sensitive than the 3σ limits (0.27% false alarm rate)
- Provides **early warning** of process shifts before they become full anomalies

## Robustness to Different Distributions

### Why XmR Charts Are Distribution-Robust

**1. Moving Range Method**: 
- Uses consecutive differences rather than population statistics
- Less sensitive to skewness, outliers, and non-normality
- The d₂ constant (1.128) remains approximately valid across distributions

**2. Empirical Evidence**:
- Donald Wheeler's research shows XmR charts maintain effectiveness with:
  - Skewed distributions
  - Heavy-tailed distributions  
  - Discrete distributions
  - Mixed distributions

**3. Central Limit Theorem Effects**:
- Moving ranges of consecutive points tend toward normality
- Even with non-normal individual values, the range statistics behave predictably

**4. Conservative Nature**:
- The method tends to be slightly conservative (fewer false alarms) with non-normal data
- This preserves the practical utility while maintaining statistical rigor

### Limitations

While robust, the method works best when:
- Data points are independent 
- The process is reasonably stable (no major trends)
- Sample size is adequate (typically 15+ points for reliable limits)

## Detection Power

The 3 of 4 rule effectively detects:
- **Process mean shifts** of 0.5-1.5σ magnitude
- **Trends** toward the limits before they breach 3σ boundaries
- **Gradual deterioration** in process control
- **Systematic changes** that develop over time

## Long Run Detection: 8 Consecutive Points on Same Side

### Definition and Purpose

A **long run** occurs when 8 or more consecutive points fall on the same side of the center line. This pattern indicates a potential sustained shift in the process mean or a systematic bias.

### Mathematical Basis

**Probability Calculation**:
Assuming a stable process where each point has a 50% probability of being above or below the center line:

```
P(8 consecutive points on one side) = (0.5)^8 = 1/256 ≈ 0.39%
```

**Why 8 Points?**
1. **Statistical Significance**: 0.39% probability means this pattern occurs by chance roughly 1 in 256 times
2. **Balance**: Fewer points (e.g., 6) would create too many false alarms (~1.6%); more points (e.g., 10) might miss shifts (~0.1%)
3. **Practical Sensitivity**: Detects process shifts of approximately 1.5σ magnitude over time

### Extended Sequences

For longer sequences, the probabilities become extremely small:
- **9 consecutive**: (0.5)^9 ≈ 0.20% (1 in 512)
- **10 consecutive**: (0.5)^10 ≈ 0.10% (1 in 1,024)
- **12 consecutive**: (0.5)^12 ≈ 0.024% (1 in 4,096)

### Detection Power

Long runs effectively detect:
- **Mean shifts** of 1-2σ magnitude
- **Systematic bias** in measurement or process
- **Gradual drift** in process parameters
- **Tool wear** or equipment deterioration

## Anomaly Detection: Points Outside 3σ Limits

### Definition and Purpose

**Anomalies** are individual points that fall outside the Natural Process Limits (NPL), indicating special cause variation requiring immediate investigation.

### Mathematical Basis

**3-Sigma Rule**:
In a normal distribution:
- **99.73%** of points fall within ±3σ
- **0.27%** of points fall outside ±3σ (0.135% in each tail)

**Probability Calculation**:
```
P(point outside ±3σ) = 1 - 0.9973 = 0.0027 = 0.27%
```

This means approximately **1 in 370** points will appear as anomalies due to random chance alone.

### Why 3σ Limits?

**Historical Development**:
1. **Walter Shewhart** (1920s) established 3σ as optimal balance between:
   - **Type I Error**: False alarms (seeing signals that aren't there)
   - **Type II Error**: Missed signals (failing to detect real changes)

2. **Economic Considerations**: 
   - 2σ limits: Too many false alarms (~5%)
   - 4σ limits: Miss too many real signals (~0.006%)
   - 3σ limits: Practical balance for industrial use

### Detection Sensitivity

**Shift Detection**:
- **1σ shift**: Detected in ~44% of subsequent points
- **2σ shift**: Detected in ~84% of subsequent points  
- **3σ shift**: Detected in ~99% of subsequent points

### XmR Specific Implementation

For XmR charts, the limits are calculated as:
```
Upper NPL = mean + 2.660 × mean_mR
Lower NPL = max(mean - 2.660 × mean_mR, 0)
```

Where the 2.660 constant converts the moving range estimate to approximate 3σ limits.

## Integration of Detection Rules

### Complementary Detection

The three detection methods work together:

1. **Anomalies** (0.27% false alarm rate): Detect large, immediate shifts
2. **Long runs** (0.39% false alarm rate): Detect sustained smaller shifts  
3. **Short runs** (0.11% false alarm rate): Detect emerging shifts early

### Combined False Alarm Rate

When all three rules are applied simultaneously, the overall false alarm rate increases but remains practical:
- **Individual rules**: 0.11% + 0.27% + 0.39% = 0.77%
- **Actual combined rate**: Slightly lower due to statistical dependence
- **Practical result**: ~1 false alarm per 130-150 stable points

## Sources and References

### Primary Statistical Sources

1. **Wheeler, D.J.** (2000). *Understanding Variation: The Key to Managing Chaos*. SPC Press.
   - Foundational text on XmR methodology and constants derivation

2. **Shewhart, W.A.** (1931). *Economic Control of Quality of Manufactured Product*. D. Van Nostrand Company.
   - Original development of 3σ control limits

3. **Western Electric Company** (1956). *Statistical Quality Control Handbook*. Western Electric Corporation.
   - Source of the "3 of 4 points" and other detection rules

### XmR Implementation References

4. **[XmR Chart Instructions - Xmrit.com](https://xmrit.com/articles/plot-xmr-chart-instructions/)**
   - Practical implementation guide and constant explanations

5. **[XmR Scaling Constants - Xmrit.com](https://xmrit.com/articles/explaining-xmr-scaling-constants/)**
   - Mathematical derivation of 2.660 and 3.268 constants

### Probability and Statistical Theory

6. **Montgomery, D.C.** (2012). *Introduction to Statistical Quality Control*, 7th Edition. John Wiley & Sons.
   - Comprehensive coverage of control chart theory and applications

7. **[Western Electric Rules - Wikipedia](https://en.wikipedia.org/wiki/Western_Electric_rules)**
   - Historical context and probability calculations for detection rules

### Robustness Research

8. **Wheeler, D.J.** (1995). *Advanced Topics in Statistical Process Control*. SPC Press.
   - Research on distribution robustness and non-normal data handling

## Conclusion

The mathematical foundations of XmR chart detection rules are based on rigorous probability theory:

- **Short runs** (0.11% false alarm rate): Early warning system for emerging changes
- **Long runs** (0.39% false alarm rate): Detection of sustained process shifts  
- **Anomalies** (0.27% false alarm rate): Identification of immediate special causes
