from plotly.graph_objects import Figure, Scatter
from math import ceil
from numpy import sum as numpy_sum, array
from spc_plotly.utils import combine_paths
from datetime import timezone as _tz
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
                    next_start = period_ranges[current_period + 1][0]
                    try:
                        xr = pd.Timestamp(x)
                        pr = pd.Timestamp(next_start)
                        if xr.tzinfo is None:
                            xr = xr.tz_localize("UTC")
                        else:
                            xr = xr.tz_convert("UTC")
                        if pr.tzinfo is None:
                            pr = pr.tz_localize("UTC")
                        else:
                            pr = pr.tz_convert("UTC")
                        if xr >= pr:
                            current_period += 1
                    except Exception:
                        # Non-datetime (numeric/categorical)
                        if x >= next_start:
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
                next_start = period_ranges[current_period + 1][0]
                try:
                    xr = pd.Timestamp(x)
                    pr = pd.Timestamp(next_start)
                    if xr.tzinfo is None:
                        xr = xr.tz_localize("UTC")
                    else:
                        xr = xr.tz_convert("UTC")
                    if pr.tzinfo is None:
                        pr = pr.tz_localize("UTC")
                    else:
                        pr = pr.tz_convert("UTC")
                    if xr >= pr:
                        current_period += 1
                except Exception:
                    if x >= next_start:
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
    periods = []

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

        # # Print a few sample points with their classifications
        # print("\nSample point classifications:")
        # for i in range(min(70, len(period_y))):
        #     print(f"Point value: {period_y[i]}")
        #     print(f"Above upper midrange: {period_y[i] > upper_midrange[i]}")
        #     print(f"Below lower midrange: {period_y[i] < lower_midrange[i]}")
        #     print("---")
        
        # Makes a true/false list of if the point is above or below the midrange
        run_test_upper = [y > um for y, um in zip(period_y, upper_midrange)] 
        run_test_lower = [y < lm for y, lm in zip(period_y, lower_midrange)]
        
        # Find short runs within this period - improved algorithm
        high_or_low = [None] * len(period_y)
        short_runs = [False] * len(period_y)
        period_num = [period_idx] * len(period_y)
        
        # REWRITTEN: Direct detection of all qualifying 4-point windows
        # Find ALL qualifying windows first, then mark ALL points in those windows
        qualifying_windows = []
        
        for i in range(len(period_y) - 3): 
            window_upper = run_test_upper[i:i+4]
            window_lower = run_test_lower[i:i+4]
            
            # Check upper window qualification
            if (window_upper[0] and sum(window_upper) >= 3 and sum(window_lower) == 0):
                qualifying_windows.append((i, i+4, "High"))
                
            # Check lower window qualification  
            elif (window_lower[0] and sum(window_lower) >= 3 and sum(window_upper) == 0):
                qualifying_windows.append((i, i+4, "Low"))
        
        # Mark ALL points in ALL qualifying windows
        for start_idx, end_idx, run_type in qualifying_windows:
            for j in range(start_idx, end_idx):
                high_or_low[j] = run_type
                short_runs[j] = True
        
        # okay try to do the current path thing here by adding the start value for the index
        # need to do the path checks here. 

        # Periods are now non-overlapping at the index level; include all elements
        high_or_low_all.extend(high_or_low)
        short_runs_all.extend(short_runs)
        periods.extend(period_num)
    
    # print(list(zip(periods, high_or_low_all, short_runs_all)))

    # Harmonize with tooltip detection to ensure coverage matches markers
    try:
        _ia, _isr, _ilr = _get_run_identifiers_for_tooltips(
            fig=fig,
            x_type=x_type,
            y_xmr_func=y_xmr_func,
            npl_upper=None,
            npl_lower=None,
            sloped=sloped,
            period_ranges=period_ranges,
            period_ranges_index=period_ranges_index,
        )
        if _ilr and len(_ilr) == len(x_values):
            long_runs_all = list(_ilr)
    except Exception:
        pass

    # SIMPLIFIED APPROACH: Build paths from detected points only.
    # Group consecutive detected points by type and period,
    # but allow a new period to continue a run if the first point of the
    # new period shares the same run-type and the previous run was still active.
    # Harmonize with tooltip detection to ensure coverage matches what users see
    try:
        _ia, _isr, _ilr = _get_run_identifiers_for_tooltips(
            fig=fig,
            x_type=x_type,
            y_xmr_func=y_xmr_func,
            npl_upper=None,
            npl_lower=None,
            sloped=sloped,
            period_ranges=period_ranges,
            period_ranges_index=period_ranges_index,
        )
        if _ilr and len(_ilr) == len(x_values):
            long_runs_all = list(_ilr)
    except Exception:
        pass

    # Harmonize detection with tooltip logic to ensure the first index of a new period
    # is included if it qualifies (e.g., 2024-03-22). We recompute the canonical mask
    # using the tooltip helper and override our local mask when lengths match.
    try:
        _ia2, _isr2, _ilr2 = _get_run_identifiers_for_tooltips(
            fig=fig,
            x_type=x_type,
            y_xmr_func=y_xmr_func,
            npl_upper=None,
            npl_lower=None,
            sloped=sloped,
            period_ranges=period_ranges,
            period_ranges_index=period_ranges_index,
        )
        if _ilr2 and len(_ilr2) == len(x_values):
            long_runs_all = list(_ilr2)
    except Exception:
        pass

    paths = []
    if any(short_runs_all):  # Only process if we have detected points
        current_path = []
        current_type = None
        current_period = None
        
        prev_period = None
        for i, (is_run, x, y, h_l, period) in enumerate(zip(short_runs_all, x_values, y_values, high_or_low_all, periods)):
            if is_run:
                # Check if we need to start a new path
                if (
                    h_l != current_type or
                    (period != current_period and not (prev_period is not None and period == prev_period + 1 and current_path and len(current_path) >= 3 and h_l == current_type))
                ):
                    # Save previous path if it meets minimum requirements
                    if current_path and len(current_path) >= 4:  # Minimum 4 points for short runs
                        paths.append(current_path)
                    # Start new path
                    current_path = [(x, y, h_l)]
                    current_type = h_l
                    current_period = period
                else:
                    # Continue current path
                    current_path.append((x, y, h_l))
            else:
                # Point not in run - end current path if it exists
                if current_path and len(current_path) >= 4:  # Minimum 4 points for short runs
                    paths.append(current_path)
                current_path = []
                current_type = None
                current_period = None
            prev_period = period
        
        # Don't forget the final path
        if current_path and len(current_path) >= 4:  # Minimum 4 points for short runs
            paths.append(current_path)
    
    # Create path strings for visualization
    path_strings = []
    for path in paths:
        path_string = ""
        # Compute a shift delta so encoded ms align with how Plotly places the point
        delta = 0
        if x_type == "date_time" and len(path):
            d0 = path[0][0]
            # ms used in our encoding (naive midnight)
            ts0 = pd.Timestamp(d0).normalize()
            ms_path0 = (ts0 - pd.Timestamp("1970-01-01")).total_seconds() * 1000
            # ms as Plotly places the scatter point (treat naive as local time, then convert to UTC)
            dt0 = pd.Timestamp(d0).to_pydatetime()
            if dt0.tzinfo is None:
                tz = pd.Timestamp.now().to_pydatetime().astimezone().tzinfo
                dt0_local = dt0.replace(tzinfo=tz)
            else:
                dt0_local = dt0
            ms_plot0 = dt0_local.astimezone(__import__('datetime').timezone.utc).timestamp() * 1000
            delta = ms_plot0 - ms_path0

        # Draw upper edge
        for i, (d, v, _) in enumerate(path):
            dd = d
            if x_type == "date_time":
                ts = pd.Timestamp(dd).normalize()
                dd = (ts - pd.Timestamp("1970-01-01")).total_seconds() * 1000 + delta
            if i == 0:
                path_string += "M {} {}".format(dd, v + shape_buffer)
            else:
                path_string += " L {} {}".format(dd, v + shape_buffer)

        # Draw lower edge
        for d, v, _ in reversed(path):
            dd = d
            if x_type == "date_time":
                ts = pd.Timestamp(dd).normalize()
                dd = (ts - pd.Timestamp("1970-01-01")).total_seconds() * 1000 + delta
            path_string += " L {} {}".format(dd, v - shape_buffer)

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
        tuple: (shapes, paths, long_runs_all) where shapes are the visualization elements,
        paths are the point data, and long_runs_all is the boolean mask (one per x) used for detection
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
            # Include equality with the centerline as being on that side to avoid
            # dropping the first qualifying point (e.g., 2024-03-22) due to rounding.
            if period_y[i] >= period_mid[i]:
                high_or_low.append("High")
            elif period_y[i] <= period_mid[i]:
                high_or_low.append("Low")
            else:
                high_or_low.append(None)
        
        # Seed runs of 8+ explicitly at the earliest qualifying index
        n = len(high_or_low)
        for i in range(0, n - 7):
            if high_or_low[i] is None:
                continue
            side = high_or_low[i]
            ok = True
            for k in range(i, i + 8):
                if high_or_low[k] != side:
                    ok = False
                    break
            if ok:
                for k in range(i, i + 8):
                    long_runs[k] = True

        # Detect runs of 8+ using contiguous same-side segments to ensure the start
        # of a qualifying segment is included (e.g., include 2024-03-22).
        i = 0
        n = len(high_or_low)
        while i < n:
            if high_or_low[i] is None:
                i += 1
                continue
            side = high_or_low[i]
            j = i + 1
            while j < n and high_or_low[j] == side:
                j += 1
            seg_len = j - i
            if seg_len >= 8:
                for k in range(i, j):
                    long_runs[k] = True
            i = j
        
        # Specific exclusions based on user requirements  
        # for i in range(1, len(period_y)):
        #     if (high_or_low[i] == "High" and high_or_low[i-1] == "Low" and long_runs[i]):
        #         # Count consecutive Low points before this
        #         consecutive_low = 0
        #         j = i - 1
        #         while j >= 0 and high_or_low[j] == "Low":
        #             consecutive_low += 1
        #             j -= 1
                
        #         # If this breaks a sequence of 5+ consecutive Low points, exclude it
        #         if consecutive_low >= 5:
        #             long_runs[i] = False
        
        # Periods are now non-overlapping at the index level; include all elements
        high_or_low_all.extend(high_or_low)
        long_runs_all.extend(long_runs)

    paths = []
    if any(long_runs_all):  # Only process if we have detected points
        current_path = []
        current_type = None
        
        prev_is_run = False
        prev_type = None
        for i, (is_run, x, y, h_l) in enumerate(zip(long_runs_all, x_values, y_values, high_or_low_all)):
            if is_run:
                # Check if we need to start a new path
                if h_l != current_type or (not prev_is_run and current_path and len(current_path) < 8):
                    # Save previous path if it meets minimum requirements
                    if current_path and len(current_path) >= 8:  # Minimum 8 points for long runs
                        paths.append(current_path)
                    # Start new path exactly at this point
                    current_path = [(x, y, h_l)]
                    current_type = h_l
                else:
                    # Continue current path
                    current_path.append((x, y, h_l))
            else:
                # Point not in run - end current path if it exists
                if current_path and len(current_path) >= 8:  # Minimum 8 points for long runs
                    paths.append(current_path)
                current_path = []
                current_type = None
            prev_is_run = is_run
            prev_type = h_l
        
        # Don't forget the final path
        if current_path and len(current_path) >= 8:  # Minimum 8 points for long runs
            paths.append(current_path)
    
    # Create path strings for visualization (apply the same alignment logic as short runs)
    path_strings = []
    for path in paths:
        path_string = ""

        def _to_ms(val):
            dt = pd.Timestamp(val).to_pydatetime()
            if dt.tzinfo is None:
                dt = dt.astimezone()  # local tz
            return dt.astimezone(_tz.utc).timestamp() * 1000.0

        # Draw upper edge
        for i, (d, v, _) in enumerate(path):
            dd = d
            if x_type == "date_time":
                dd = _to_ms(d)
            if i == 0:
                path_string += "M {} {}".format(dd, v + shape_buffer)
            else:
                path_string += " L {} {}".format(dd, v + shape_buffer)

        # Draw lower edge
        for d, v, _ in reversed(path):
            dd = d
            if x_type == "date_time":
                dd = _to_ms(d)
            path_string += " L {} {}".format(dd, v - shape_buffer)

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

    return shapes, paths, long_runs_all


def _get_run_identifiers_for_tooltips(
    fig: Figure,
    x_type: str,
    y_xmr_func: float | list,
    npl_upper: float | list,
    npl_lower: float | list,
    sloped: bool,
    period_ranges: list = None,
    period_ranges_index: list = None,
    long_runs_mask: list | None = None,
) -> tuple:
    """
    Calculate boolean arrays identifying which points are part of long runs and short runs
    for use in tooltips. This function consolidates the logic that was previously scattered
    in the main chart method.

    Parameters:
        fig (Figure): Passed in Figure object
        x_type (str): Type of x-axis values ('date_time', 'categorical', or 'numeric')
        y_xmr_func (float|list): Center line values (mean or median)
        npl_upper (float|list): Upper natural process limits
        npl_lower (float|list): Lower natural process limits
        sloped (bool): Whether to use sloping approach for limit values
        period_ranges (list): List of period range tuples
        period_ranges_index (list): List of index range tuples for periods

    Returns:
        tuple: (is_anomaly, is_short_run, is_long_run) where each is a list of booleans
    """
    fig_data = fig.data
    x_values = [el for el in fig_data[0].x]
    y_values = fig_data[0].y

    if x_type == "date_time":
        x_values = pd.to_datetime(x_values)

    # Make sure everything has a period_ranges_index for ease of processing
    if not period_ranges_index:
        period_ranges_index = [(0, len(x_values) - 1)]

    # Initialize boolean arrays
    is_anomaly = [False] * len(x_values)
    is_short_run = [False] * len(x_values)
    is_long_run = [False] * len(x_values)

    # Get anomaly information from the figure traces
    # Anomalies are stored in traces 2 and 3 (invisible traces added by _anomalies function)
    if len(fig_data) > 2:
        # Individual values anomalies (trace index 2)
        if hasattr(fig_data[2], 'x') and fig_data[2].x:
            for anomaly_x in fig_data[2].x:
                for i, x_val in enumerate(x_values):
                    if x_val == anomaly_x:
                        is_anomaly[i] = True
                        break

        # Moving range anomalies (trace index 3)
        if len(fig_data) > 3 and hasattr(fig_data[3], 'x') and fig_data[3].x:
            for anomaly_x in fig_data[3].x:
                for i, x_val in enumerate(x_values):
                    if x_val == anomaly_x:
                        is_anomaly[i] = True
                        break

    # Calculate short run identifiers
    if period_ranges_index:
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

            # Calculate midranges for short run detection
            upper_midrange = [mid + ((upper - mid) / 2) for mid, upper in zip(period_mid, period_upper)]
            lower_midrange = [mid - ((mid - lower) / 2) for mid, lower in zip(period_mid, period_lower)]

            # Determine if points are above/below midrange
            run_test_upper = [y > um for y, um in zip(period_y, upper_midrange)]
            run_test_lower = [y < lm for y, lm in zip(period_y, lower_midrange)]

            # Find short runs within this period
            for i in range(len(period_y) - 3):
                window_upper = run_test_upper[i:i+4]
                window_lower = run_test_lower[i:i+4]

                # Check for qualifying windows
                if (window_upper[0] and sum(window_upper) >= 3 and sum(window_lower) == 0) or \
                   (window_lower[0] and sum(window_lower) >= 3 and sum(window_upper) == 0):
                    # Mark all points in this window as short run points
                    for j in range(4):
                        is_short_run[start_idx + i + j] = True

    # Calculate long run identifiers
    # If a mask was provided (from _long_run_test.long_runs_all), trust it verbatim
    if long_runs_mask is not None and len(long_runs_mask) == len(x_values):
        is_long_run = list(long_runs_mask)
    else:
        if period_ranges_index:
            for period_idx, (start_idx, end_idx) in enumerate(period_ranges_index):
                period_x = x_values[start_idx:end_idx + 1]
                period_y = y_values[start_idx:end_idx + 1]

                # Get comparison value for this period
                if sloped:
                    period_mid = [el[1] for el in y_xmr_func[period_idx]]
                else:
                    if isinstance(y_xmr_func, list):
                        period_mid = [y_xmr_func[period_idx]] * len(period_y)
                    else:
                        period_mid = [y_xmr_func] * len(period_y)

                # Determine if points are above/below midline
                high_or_low = []
                for i in range(len(period_y)):
                    if period_y[i] > period_mid[i]:
                        high_or_low.append("High")
                    elif period_y[i] < period_mid[i]:
                        high_or_low.append("Low")
                    else:
                        high_or_low.append(None)

                # Detect runs of 8 or more consecutive points on same side
                for i in range(len(high_or_low) - 7):
                    window = high_or_low[i:i+8]
                    if all(x == window[0] for x in window) and window[0] is not None:
                        for j in range(8):
                            is_long_run[start_idx + i + j] = True

    return is_anomaly, is_short_run, is_long_run


def _build_long_run_shapes_from_flags(
    fig: Figure,
    is_long_run: list,
    y_xmr_func,
    x_type: str,
    sloped: bool,
    fill_color: str = "pink",
    line_color: str = "purple",
    line_width: int = 2,
    line_type: str = "longdashdot",
    opacity: float = 0.2,
    shape_buffer_pct: float = 0.05,
    period_ranges_index: list = None,
) -> tuple:
    """
    Build long-run highlight shapes directly from a provided boolean mask (is_long_run).

    Parameters:
        fig (Figure): Figure containing the base traces
        is_long_run (list[bool]): Per-point flags indicating long-run membership
        y_xmr_func (float|list): Center line values per period
        x_type (str): X-axis type
        sloped (bool): Whether sloped midlines are used
        fill_color (str): Fill color
        line_color (str): Border color
        line_width (int): Border width
        line_type (str): Border dash type
        opacity (float): Fill opacity
        shape_buffer_pct (float): Vertical buffer percent for shape thickness
        period_ranges_index (list): Period index ranges

    Returns:
        tuple: (shapes, paths)
    """
    fig_data = fig.data
    x_values = [el for el in fig_data[0].x]
    y_values = fig_data[0].y
    y_range = fig.layout.yaxis.range
    shape_buffer = (y_range[1] - y_range[0]) * shape_buffer_pct

    if x_type == "date_time":
        x_values = pd.to_datetime(x_values)

    if not period_ranges_index:
        period_ranges_index = [(0, len(x_values) - 1)]

    # Normalize midlines per period to compute side (High/Low)
    if not isinstance(y_xmr_func, list):
        y_xmr_func = [y_xmr_func] * len(x_values)

    high_or_low_all = []
    for period_idx, (start_idx, end_idx) in enumerate(period_ranges_index):
        period_y = y_values[start_idx:end_idx + 1]
        if sloped:
            period_mid = [el[1] for el in y_xmr_func[period_idx]]
        else:
            period_mid = [y_xmr_func[period_idx]] * len(period_y)
        for i in range(len(period_y)):
            if period_y[i] >= period_mid[i]:
                high_or_low_all.append("High")
            else:
                high_or_low_all.append("Low")

    # Group consecutive True flags ONLY (ignore side changes) and build paths
    paths = []
    current_path = []
    for flag, x, y, h_l in zip(is_long_run, x_values, y_values, high_or_low_all):
        if flag:
            current_path.append((x, y, h_l))
        else:
            if current_path and len(current_path) >= 8:
                paths.append(current_path)
            current_path = []
    if current_path and len(current_path) >= 8:
        paths.append(current_path)

    # Encode paths into Plotly shape paths
    def _to_ms(val):
        dt = pd.Timestamp(val).to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.astimezone()
        return dt.astimezone(_tz.utc).timestamp() * 1000.0

    path_strings = []
    for path in paths:
        path_string = ""
        for i, (d, v, _) in enumerate(path):
            dd = _to_ms(d) if x_type == "date_time" else d
            if i == 0:
                path_string += f"M {dd} {v + shape_buffer}"
            else:
                path_string += f" L {dd} {v + shape_buffer}"
        for d, v, _ in reversed(path):
            dd = _to_ms(d) if x_type == "date_time" else d
            path_string += f" L {dd} {v - shape_buffer}"
        path_string += " Z"
        path_strings.append(path_string)

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


def _build_short_run_shapes_from_flags(
    fig: Figure,
    is_short_run: list,
    x_type: str,
    fill_color: str = "purple",
    line_color: str = "blue",
    line_width: int = 2,
    line_type: str = "longdashdot",
    opacity: float = 0.2,
    shape_buffer_pct: float = 0.05,
) -> tuple:
    """
    Build short-run highlight shapes directly from a provided boolean mask (is_short_run).

    Parameters:
        fig (Figure): Figure containing the base traces
        is_short_run (list[bool]): Per-point flags indicating short-run membership
        x_type (str): X-axis type
        fill_color (str): Fill color
        line_color (str): Border color
        line_width (int): Border width
        line_type (str): Border dash type
        opacity (float): Fill opacity
        shape_buffer_pct (float): Vertical buffer percent for shape thickness

    Returns:
        tuple: (shapes, paths)
    """
    fig_data = fig.data
    x_values = [el for el in fig_data[0].x]
    y_values = fig_data[0].y
    y_range = fig.layout.yaxis.range
    shape_buffer = (y_range[1] - y_range[0]) * shape_buffer_pct

    if x_type == "date_time":
        x_values = pd.to_datetime(x_values)

    # Group contiguous True blocks, min length 4
    paths = []
    current_path = []
    for flag, x, y in zip(is_short_run, x_values, y_values):
        if flag:
            current_path.append((x, y, None))
        else:
            if current_path and len(current_path) >= 4:
                paths.append(current_path)
            current_path = []
    if current_path and len(current_path) >= 4:
        paths.append(current_path)

    def _to_ms(val):
        dt = pd.Timestamp(val).to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.astimezone()
        return dt.astimezone(_tz.utc).timestamp() * 1000.0

    path_strings = []
    for path in paths:
        path_string = ""
        for i, (d, v, _) in enumerate(path):
            dd = _to_ms(d) if x_type == "date_time" else d
            if i == 0:
                path_string += f"M {dd} {v + shape_buffer}"
            else:
                path_string += f" L {dd} {v + shape_buffer}"
        for d, v, _ in reversed(path):
            dd = _to_ms(d) if x_type == "date_time" else d
            path_string += f" L {dd} {v - shape_buffer}"
        path_string += " Z"
        path_strings.append(path_string)

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
