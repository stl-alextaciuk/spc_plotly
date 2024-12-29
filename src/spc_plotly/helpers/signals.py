from plotly.graph_objects import Figure, Scatter
from math import ceil
from numpy import sum as numpy_sum, array
from spc_plotly.utils import combine_paths
import pandas as pd

def _anomalies(
    fig: Figure,
    npl_upper: float | list,
    npl_lower: float | list,
    mR_upper: float,
    sloped: bool,
    period_ranges: list = None
) -> tuple:
    """
    Identifies all points that lie outside of the natural process limits

    Parameters:
        fig (Figure): Passed in Figure object
        npl_upper (float|list): Upper process limit. If sloped is True, this is a list of tuples.
        npl_lower (float|list): Lower process limit. If sloped is True, this is a list of tuples.
        mR_upper (float): Upper moving range limit.
        sloped (bool): Use sloping approach for limit values.

    Returns:
        Figure: Passed in Figure object with added traces for anomalous points
        list[tuple]: All points that lie outside of the limits -> (x-value, y-value, "High"|"Low")
    """
    # Detect point outside of natural limits

    fig_data = fig.data

    if sloped:
        anomaly_points_raw = [
            (x, y, "High" if y >= upper[1] else None, "Low" if y <= lower[1] else None)
            for x, y, upper, lower in zip(
                fig_data[0].x, fig_data[0].y, npl_upper, npl_lower
            )
        ]
        anomaly_points = [
            (x, y, upper if upper is not None else lower)
            for (x, y, upper, lower) in anomaly_points_raw
            if not (upper is None and lower is None)
        ]
    else:
        if isinstance(npl_upper, list):
            # Handle multiple periods
            anomaly_points = []
            current_period = 0
            for x, y in zip(fig_data[0].x, fig_data[0].y):
                # Find which period this point belongs to
                if period_ranges and current_period < len(period_ranges) - 1:
                    if x >= period_ranges[current_period + 1][0]:
                        current_period += 1
                
                if y >= npl_upper[current_period] or y <= npl_lower[current_period]:
                    anomaly_points.append(
                        (x, y, "High" if y >= npl_upper[current_period] else "Low")
                    )
        else:
            anomaly_points = [
                (x, y, "High" if y >= npl_upper else "Low")
                for x, y in zip(fig_data[0].x, fig_data[0].y)
                if y >= npl_upper or y <= npl_lower
            ]

    fig.add_trace(
        Scatter(
            x=[x[0] for x in anomaly_points],
            y=[x[1] for x in anomaly_points],
            texttemplate="%{y}",
            mode="markers",
            marker=dict(size=8, color="red", symbol="cross"),
            visible=False,
        ),
        row=1,
        col=1,
    )

    # Handle mR anomalies
    if isinstance(mR_upper, list):
        mR_anomaly_points = []
        current_period = 0
        for x, y in zip(fig_data[1].x, fig_data[1].y):
            if period_ranges and current_period < len(period_ranges) - 1:
                if x >= period_ranges[current_period + 1][0]:
                    current_period += 1
            
            if y >= mR_upper[current_period]:
                mR_anomaly_points.append((x, y))
    else:
        mR_anomaly_points = [
            (x, y) for x, y in zip(fig_data[1].x, fig_data[1].y) if y >= mR_upper
        ]

    fig.add_trace(
        Scatter(
            x=[x[0] for x in mR_anomaly_points],
            y=[x[1] for x in mR_anomaly_points],
            mode="markers",
            marker=dict(size=8, color="red", symbol="cross"),
            visible=False,
        ),
        row=2,
        col=1,
    )

    return fig, anomaly_points


def _short_run_test(
    fig: Figure,
    npl_upper: float | list,
    npl_lower: float | list,
    y_xmr_func: float | list,
    x_type: str,
    sloped: bool,
    fill_color: str = "purple",
    line_color: str = "blue",
    line_width: int = 2,
    line_type: str = "longdashdot",
    opacity: float = 0.2,
    shape_buffer_pct: float = 0.05,
    period_ranges: list = None,
    period_ranges_index: list = None,
) -> tuple:
    fig_data = fig.data
    x_values = [el for el in fig_data[0].x]
    y_values = fig_data[0].y
    y_range = fig.layout.yaxis.range
    shape_buffer = (y_range[1] - y_range[0]) * shape_buffer_pct

    if x_type == "date_time":
        x_values = pd.to_datetime(x_values)

    # make sure everything has a period_ranges_index for ease of processing
    if not period_ranges_index:
        period_ranges_index = [(0, len(x_values) - 1)]

    # Handle different period scenarios
    high_or_low_all = []
    short_runs_all = []
    
    # Process each period separately
    for period_idx, (start_idx, end_idx) in enumerate(period_ranges_index):
        period_x = x_values[start_idx:end_idx + 1]
        period_y = y_values[start_idx:end_idx + 1]
        
        # Get comparison values for this period
        if sloped:
            period_mid = [el[1] for el in y_xmr_func[period_idx]]
            period_upper = [el[1] for el in npl_upper[period_idx]]
            period_lower = [el[1] for el in npl_lower[period_idx]]
        else:
            # Handle both scalar and list inputs
            if isinstance(y_xmr_func, list):
                period_mid = [y_xmr_func[period_idx]] * len(period_y)
                period_upper = [npl_upper[period_idx]] * len(period_y)
                period_lower = [npl_lower[period_idx]] * len(period_y)
            else:
                period_mid = [y_xmr_func] * len(period_y)
                period_upper = [npl_upper] * len(period_y)
                period_lower = [npl_lower] * len(period_y)
        
        # Calculate midranges
        upper_midrange = [mid + ((upper - mid) / 2) for mid, upper in zip(period_mid, period_upper)]
        lower_midrange = [mid - ((mid - lower) / 2) for mid, lower in zip(period_mid, period_lower)]
        
        # Find runs within this period
        high_or_low = []
        short_runs = [False]*len(period_y)
        
        # Determine if points are closer to limits than midline
        for i in range(len(period_y)):
            if period_y[i] > upper_midrange[i]:
                high_or_low.append("High")
            elif period_y[i] < lower_midrange[i]:
                high_or_low.append("Low")
            else:
                high_or_low.append(None)
        
        # Detect 3 out of 4 consecutive points
        for i in range(len(high_or_low) - 3):
            window = high_or_low[i:i+4]
            high_count = sum(1 for x in window if x == "High")
            low_count = sum(1 for x in window if x == "Low")
            if high_count >= 3 or low_count >= 3:
                short_runs[i:i+4] = [True]*4
        
        high_or_low_all += high_or_low
        short_runs_all += short_runs
    
    # Build paths from runs
    paths = []
    current_path = []
    current_type = None
    
    # Combine points into runs
    for i, (is_run, x, y, h_l) in enumerate(zip(short_runs_all, x_values, y_values, high_or_low_all)):
        if is_run:
            if not current_path:  # Start of a new run
                current_type = h_l
                # Look back to include previous 3 points if they match
                start_idx = max(0, i-3)
                if sum(1 for h in high_or_low_all[start_idx:i] if h == h_l) >= 2:
                    current_path.extend([
                        (x_values[j], y_values[j], high_or_low_all[j])
                        for j in range(start_idx, i)
                    ])
            if h_l == current_type:  # Continue current run
                current_path.append((x, y, h_l))
            else:  # Type changed, end current run
                if len(current_path) >= 3:
                    paths.append(current_path)
                current_path = [(x, y, h_l)]
                current_type = h_l
        elif current_path:  # End of a run
            if len(current_path) >= 3:
                paths.append(current_path)
            current_path = []
            current_type = None
    
    # Add final path if it exists
    if current_path and len(current_path) >= 3:
        paths.append(current_path)
    
    # Create path strings for visualization
    path_strings = []
    for path in paths:
        path_string = ""
        # Draw upper edge
        for i, (d, v, _) in enumerate(path):
            # Convert date to numeric value if needed
            if x_type == "date_time":
                d = pd.Timestamp(d).timestamp() * 1000  # Convert to milliseconds
            
            if i == 0:
                path_string += "M {} {}".format(d, v + shape_buffer)
            else:
                path_string += " L {} {}".format(d, v + shape_buffer)
        
        # Draw lower edge
        for d, v, _ in reversed(path):
            # Convert date to numeric value if needed
            if x_type == "date_time":
                d = pd.Timestamp(d).timestamp() * 1000  # Convert to milliseconds
            path_string += " L {} {}".format(d, v - shape_buffer)
        
        path_string += " Z"
        path_strings.append(path_string)
    
    # Create shapes for visualization
    shapes = []
    for path_string in path_strings:
        shapes.append({
            "fillcolor": fill_color,
            "line": {"color": line_color, "dash": line_type, "width": line_width},
            "name": "Short Run",
            "opacity": opacity,
            "path": path_string,
            "type": "path",
        })
    
    return shapes, paths


def _long_run_test(
    fig: Figure,
    y_xmr_func,
    x_type: str,
    sloped: bool,
    fill_color: str = "pink",
    line_color: str = "purple",
    line_width: int = 2,
    line_type: str = "longdashdot",
    opacity: float = 0.2,
    shape_buffer_pct: float = 0.05,
    period_ranges: list = None,
    period_ranges_index: list = None,
) -> tuple:
    """
    Identifies "long runs", defined as 8 consecutive points above or below the mid line.

    Parameters:
        fig (Figure): Passed in Figure object
        y_xmr_func (float|list): Center line values (mean or median). If sloped is True or using period ranges, 
            this is a list of values or tuples.
        x_type (str): Type of x-axis values ('date_time', 'categorical', or 'numeric')
        sloped (bool): Whether to use sloping approach for limit values
        fill_color (str): Fill color for highlighting runs
        line_color (str): Line color for run borders
        line_width (int): Width of border lines
        line_type (str): Type of border lines
        opacity (float): Opacity of fill color
        shape_buffer_pct (float): Buffer percentage for shape visualization
        period_ranges (list): List of period range tuples
        period_ranges_index (list): List of index range tuples for periods

    Returns:
        tuple: (shapes, paths) where shapes are the visualization elements and paths are the point data
    """
    fig_data = fig.data
    x_values = [el for el in fig_data[0].x]
    y_values = fig_data[0].y
    y_range = fig.layout.yaxis.range
    shape_buffer = (y_range[1] - y_range[0]) * shape_buffer_pct

    if x_type == "date_time":
        x_values = pd.to_datetime(x_values)

    # make sure everything has a period_ranges_index for ease of processing
    if not period_ranges_index:
        period_ranges_index = [(0, len(x_values) - 1)]

    # same logic, need to make this into a list as needed
    if not isinstance(y_xmr_func, list):
        y_xmr_func = [y_xmr_func] * len(x_values)

    # Handle different period scenarios

    high_or_low_all = []
    long_runs_all = []
    
    # Process each period separately
    for period_idx, (start_idx, end_idx) in enumerate(period_ranges_index):
        period_x = x_values[start_idx:end_idx + 1]
        period_y = y_values[start_idx:end_idx + 1]
        
        # Get comparison value for this period
        if sloped:
            period_mid = [el[1] for el in y_xmr_func[period_idx]]
        else:
            period_mid = [y_xmr_func[period_idx]] * len(period_y)
        
        # Find runs within this period
        high_or_low = []
        long_runs = [False]*len(period_y)
        
        # Determine if points are above/below midline
        for i in range(len(period_y)):
            if period_y[i] > period_mid[i]:
                high_or_low.append("High")
            elif period_y[i] < period_mid[i]:
                high_or_low.append("Low")
            else:
                high_or_low.append(None)
        
        # Detect runs of 8 or more
        for i in range(len(high_or_low) - 7):
            window = high_or_low[i:i+8]
            if all(x == window[0] for x in window) and window[0] is not None:
                long_runs[i:i+8] = [True]*8
        
        high_or_low_all.extend(high_or_low)
        long_runs_all.extend(long_runs)
    
    # Build paths from runs
    paths = []
    current_path = []
    current_type = None
    
    # Combine points into runs
    for i, (is_run, x, y, h_l) in enumerate(zip(long_runs_all, x_values, y_values, high_or_low_all)):
        if is_run:
            if not current_path:  # Start of a new run
                current_type = h_l
                # Look back to include previous 7 points if they match
                start_idx = max(0, i-7)
                if all(h == h_l for h in high_or_low_all[start_idx:i]):
                    current_path.extend([
                        (x_values[j], y_values[j], high_or_low_all[j])
                        for j in range(start_idx, i)
                    ])
            if h_l == current_type:  # Continue current run
                current_path.append((x, y, h_l))
            else:  # Type changed, end current run
                if len(current_path) >= 8:
                    paths.append(current_path)
                current_path = [(x, y, h_l)]
                current_type = h_l
        elif current_path:  # End of a run
            if len(current_path) >= 8:
                paths.append(current_path)
            current_path = []
            current_type = None
    
    # Add final path if it exists
    if current_path and len(current_path) >= 8:
        paths.append(current_path)
    
    # Create path strings for visualization
    path_strings = []
    for path in paths:
        path_string = ""
        # Draw upper edge
        for i, (d, v, _) in enumerate(path):
            # Convert date to numeric value if needed
            if x_type == "date_time":
                d = pd.Timestamp(d).timestamp() * 1000  # Convert to milliseconds
            
            if i == 0:
                path_string += "M {} {}".format(d, v + shape_buffer)
            else:
                path_string += " L {} {}".format(d, v + shape_buffer)
        
        # Draw lower edge
        for d, v, _ in reversed(path):
            # Convert date to numeric value if needed
            if x_type == "date_time":
                d = pd.Timestamp(d).timestamp() * 1000  # Convert to milliseconds
            path_string += " L {} {}".format(d, v - shape_buffer)
        
        path_string += " Z"
        path_strings.append(path_string)
    
    # Create shapes for visualization
    shapes = []
    for path_string in path_strings:
        shapes.append({
            "fillcolor": fill_color,
            "line": {"color": line_color, "dash": line_type, "width": line_width},
            "name": "Long Run",
            "opacity": opacity,
            "path": path_string,
            "type": "path",
        })
    
    return shapes, paths
