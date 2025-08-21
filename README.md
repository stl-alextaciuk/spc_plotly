# spc_plotly

**spc_plotly** is a Python helper library for creating XmR Charts according to the theories of Statistical Process Control using the Plotly library.

XmR Charts allow the viewer to quickly identify signals in a data set and ignore routine variation. The library provides advanced signal detection capabilities including anomaly detection, short runs, and long runs with support for period-based analysis and dynamic limit calculations.

## Installation

```shell
pip install spc-plotly
```

## Usage

```python
from spc_plotly import xmr
import pandas as pd

counts = [
    2478, 2350, 2485, 2296, 2359, 2567, 3089, 2668, 1788, 2854, 
    2365, 1883, 1959, 1927, 2640, 2626, 2144, 2409, 4412, 3287, 
    3049, 3364, 3078, 2972, 3415, 2753, 3102, 2191, 3693, 4385, 
    4699, 3031, 2659, 3885, 2628, 2621, 3071, 2002
]
periods = [
    '2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', 
    '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12',
    '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06',
    '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12', 
    '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06',
    '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12',
    '2024-01', '2024-02'
]
data = pd.DataFrame({
    "Period": periods,
    "Count": counts
})

xmr_chart = xmr.XmR(
    data=data,
    x_ser_name="Period",
    y_ser_name="Count",
    x_cutoff="2023-06",
    date_part_resolution="month", # This should match your data
    custom_date_part="",
    xmr_function="mean"
)

xmr_chart.mR_limit_values
# {
#   'mR_xmr_func': 571.8918918918919,
#   'mR_upper_limit': 1868.9427027027025,
#   'xmr_func': 'mean'
# }
xmr_chart.npl_limit_values
# {
#   'y_xmr_func': 2820.6315789473683,
#   'npl_upper_limit': 4341.864011379801,
#   'npl_lower_limit': 1299.3991465149359,
#   'xmr_func': 'mean'
# }
xmr_chart.signals
# {'anomalies': [(datetime.datetime(2022, 7, 1, 0, 0), 4412, 'High'),
#   (datetime.datetime(2023, 6, 1, 0, 0), 4385, 'High'),
#   (datetime.datetime(2023, 7, 1, 0, 0), 4699, 'High')],
#  'long_runs': [[('2021-11', 2365, 'Low'),
#    ('2021-12', 1883, 'Low'),
#    ('2022-01', 1959, 'Low'),
#    ('2022-02', 1927, 'Low'),
#    ('2022-03', 2640, 'Low'),
#    ('2022-04', 2626, 'Low'),
#    ('2022-05', 2144, 'Low'),
#    ('2022-06', 2409, 'Low')]],
#  'short_runs': [[('2023-04', 2191, 'High'),
#    ('2023-05', 3693, 'High'),
#    ('2023-06', 4385, 'High'),
#    ('2023-07', 4699, 'High'),
#    ('2023-08', 3031, 'High')],
#   [('2021-11', 2365, 'Low'),
#    ('2021-12', 1883, 'Low'),
#    ('2022-01', 1959, 'Low'),
#    ('2022-02', 1927, 'Low'),
#    ('2022-03', 2640, 'Low')]]}

xmr_chart.xmr_chart
```
### Example gif from original implementation of this library
<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMGN2d3p3cG1heG90OGZyb2tzeWZsYmp6eXZmajd5MHJqcmhwczZwNCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/J8ECUUcN5WWwYR5vph/giphy.gif" width="500" />
<!-- ![XmR Example]() -->

## Data Types

The library supports three types of x-axis data:

### Date-Time Data
Use `x_type="date_time"` for temporal data. Specify the resolution with `date_part_resolution`:

```python
# Monthly data
xmr_chart = xmr.XmR(
    data=data,
    x_ser_name="Period",
    y_ser_name="Count", 
    x_type="date_time",
    date_part_resolution="month"  # year, month, day, hour, minute, or custom
)
```

### Numeric Data
Use `x_type="numeric"` for ordered numerical sequences:

```python
# Sequential numeric data
data = pd.DataFrame({
    "sequence": [1, 2, 3, 4, 5, 6],
    "values": [100, 102, 98, 105, 99, 101]
})

xmr_chart = xmr.XmR(
    data=data,
    x_ser_name="sequence", 
    y_ser_name="values",
    x_type="numeric"
)
```

### Categorical Data
Use `x_type="categorical"` for ordered categories:

```python
# Categorical data (e.g., stations, products, regions)
data = pd.DataFrame({
    "station": ["Station_A", "Station_B", "Station_C", "Station_D"],
    "quality_score": [85, 87, 82, 90]
})

xmr_chart = xmr.XmR(
    data=data,
    x_ser_name="station",
    y_ser_name="quality_score", 
    x_type="categorical"
)
```


## Signal Detection

The library automatically detects three types of signals in your data:

### Anomalies
Points that fall outside the natural process limits (NPL). These represent special cause variation that requires investigation.

### Short Runs
Sequences where 3 out of 4 consecutive points are closer to the natural limits than to the center line. Requirements:
- All four points must be in the same period (when using period breaks)
- At least 3 of the 4 points must be closer to the same limit (upper or lower)
- The first point in the sequence must be closer to a natural limit than the center line
- Uses period-specific limit calculations

### Long Runs  
Sequences of 8 or more consecutive points on the same side of the center line. Requirements:
- All points must be on the same side of the center line (above or below)
- All points must be in the same period (when using period breaks)
- Uses period-specific limit calculations

## Period-Based Analysis

### Period Breaks
Use `period_breaks` to recalculate limits for specific ranges of your data:

```python
xmr_chart = xmr.XmR(
    data=data,
    x_ser_name="Period", 
    y_ser_name="Count",
    period_breaks=["2022-06", "2023-01"],  # Recalculate limits at these points
    date_part_resolution="month",
    xmr_function="mean"
)
```

### Period End
Use `period_end` to fix the calculation point for the final period:

```python
xmr_chart = xmr.XmR(
    data=data,
    x_ser_name="Period",
    y_ser_name="Count", 
    period_end="2023-06",  # Use data up to this point for final period limits
    date_part_resolution="month",
    xmr_function="mean"
)
```

### Combined Usage
Combine both features for maximum control:

```python
xmr_chart = xmr.XmR(
    data=data,
    x_ser_name="Period",
    y_ser_name="Count",
    period_breaks=["2022-06"],  # Create period break
    period_end="2023-06",       # Fix final period calculation point  
    date_part_resolution="month",
    xmr_function="mean"
)
```

For reference, please read [Making Sense of Data by Donald Wheeler](https://www.amazon.com/Making-Sense-Data-Donald-Wheeler/dp/0945320728) and [Twenty Things You Need To Know](https://www.amazon.com/Twenty-Things-You-Need-Know/dp/094532068X)

### Use the Median

If your data contains extreme outliers, you can update the xmr_function parameter to "median"

```python
xmr_chart = xmr.XmR(
    data=data,
    x_ser_name="Period",
    y_ser_name="Count",
    x_cutoff="2023-06",
    xmr_function="median"
    sloped=False
)
```

### Sloped Limits

Coming soon.

### Alternative method: Calculate Limits from Subset of Data

This is a legacy method of splitting data into different periods - I recommend using the `period_breaks` and `period_end` instead. 
If you want to calculate the limits from a subset of your data, use the x_cutoff and x_begin parameters.

```python
xmr_chart = xmr.XmR(
    data=data,
    x_ser_name="Period",
    y_ser_name="Count",
    x_begin="2022-01",
    x_cutoff="2023-06",
    xmr_function="median"
    sloped=False
)
```
These paremeters are *inclusive*, so they will include all data between "2022-01" and "2023-06". If no value is passed, `x_begin` and `x_cutoff` will be set to the minimum and maximum values, respectively.

### Resources and Statistical Significance

The three signal patterns detected by this library have deep statistical foundations in Statistical Process Control (SPC) theory:

**Anomalies** represent special cause variation - points that fall outside the natural process limits have less than a 0.3% probability of occurring due to common cause variation alone. When detected, they indicate that something fundamentally different happened in your process that requires investigation and action.

**Long Runs** (8+ consecutive points on one side of the center line) have approximately a 0.4% probability of occurring by chance in a stable process [source](https://www.staceybarr.com/measure-up/3-essential-signals-to-look-for-in-your-kpis/#:~:text=SIGNAL%202:%20Long%20run,are%20to%20the%20Central%20Line.). As [Commoncog notes](https://commoncog.com/becoming-data-driven-first-principles/), these patterns help you distinguish between routine variation and meaningful process shifts, enabling you to "pursue knowledge" rather than react to random fluctuations.

**Short Runs** (3 of 4 points closer to limits than center) indicate early process drift and are derived from the [Western Electric Rules](https://en.wikipedia.org/wiki/Western_Electric_rules) established in the 1950s for detecting non-random patterns in control charts. This pattern provides earlier detection of process changes before they become full anomalies, following Wheeler and Deming's principle that understanding variation is key to process improvement.

These statistical thresholds, developed through decades of industrial application, help organizations avoid the two fundamental errors in data interpretation: seeing signals where none exist (Type I error) and missing real signals in the noise (Type II error). As documented in the [XMR manual](https://xmrit.com/manual/), this approach transforms raw data into actionable knowledge by separating predictable variation from meaningful change.

For deeper understanding: [Making Sense of Data by Donald Wheeler](https://www.amazon.com/Making-Sense-Data-Donald-Wheeler/dp/0945320728), [Twenty Things You Need To Know](https://www.amazon.com/Twenty-Things-You-Need-Know/dp/094532068X), and [Commoncog's series on becoming data-driven](https://commoncog.com/becoming-data-driven-first-principles/). 

## Dependencies
Plotly, Pandas, and Numpy
