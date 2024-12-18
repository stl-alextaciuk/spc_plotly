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
) -> tuple:
    """
    Identifies "short runs", defined as 3 out of 4 consecutive points closer to a limit
        line than the mid line.

    Parameters:
        fig (Figure): Passed in Figure object
        npl_upper (float|list): Upper process limit. If sloped is True, this is a list of tuples.
        npl_lower (float|list): Lower process limit. If sloped is True, this is a list of tuples.
        y_xmr_func (float|list): Upper moving range limit. If sloped is True, this is a list of tuples.
        sloped (bool): Use sloping approach for limit values.
        fill_color (str): Fill color of shape
        line_color (str): Line color of shape
        line_width (int): Line width of shape border
        line_type (str): Line type of shape border
        opacity (float): Opacity of shape fill
        shape_buffer_pct (float): % buffer to use for shape build. For example:
            If y-value = 100, a shape buffer of 5% would mean the lower and upper values
            of the shape for that y-value are [95, 105].

    Returns:
        list[dict]: List of dictionaries that represent a shape for each short run that will
            highlight the area of the chart containing the short run.
        list[list]: List of lists, each sublist is a path, containing a tuple that represents
            each point in the long run.
    """
    # Detect 8 consecutive points on one side of center line
    fig_data = fig.data
    x_values = [el for el in fig_data[0].x]
    y_values = fig_data[0].y

    # Convert dates if needed
    if x_type == "date_time":
        x_values = pd.to_datetime(x_values)
        
    short_runs = []
    y_range = fig.layout.yaxis.range

    if sloped:
        upper_midrange = [
            (mid[0], mid[1] + ((upper[1] - mid[1]) / 2))
            for (upper, mid) in zip(npl_upper, y_xmr_func)
        ]
        lower_midrange = [
            (mid[0], mid[1] - ((mid[1] - lower[1]) / 2))
            for (lower, mid) in zip(npl_lower, y_xmr_func)
        ]
        run_test_upper = y_values > [el[1] for el in upper_midrange]
        run_test_lower = y_values < [el[1] for el in lower_midrange]

        shape_buffer = (y_range[1] - y_range[0]) * shape_buffer_pct

        for i, el in enumerate(y_values):
            min_i = max(0, i - 3)
            max_i = i + 1
            trailing_sum_upper = numpy_sum(run_test_upper[min_i:max_i])
            trailing_sum_lower = numpy_sum(run_test_lower[min_i:max_i])
            if trailing_sum_upper >= 3:
                short_runs.append(
                    zip(
                        x_values[min_i:max_i],
                        y_values[min_i:max_i],
                        ["High"] * (max_i - min_i),
                    )
                )
            elif trailing_sum_lower >= 3:
                short_runs.append(
                    zip(
                        x_values[min_i:max_i],
                        y_values[min_i:max_i],
                        ["Low"] * (max_i - min_i),
                    )
                )
            else:
                pass

    else:
        # Handle both single and multiple periods
        if isinstance(y_xmr_func, list):
            # Create arrays for comparison based on period ranges
            y_func_values = []
            upper_midrange_values = []
            lower_midrange_values = []
            current_period = 0
            
            for x in x_values:
                if period_ranges and current_period < len(period_ranges) - 1:
                    if x >= period_ranges[current_period + 1][0]:
                        current_period += 1
                        
                curr_y_xmr = y_xmr_func[current_period]
                curr_upper = npl_upper[current_period]
                curr_lower = npl_lower[current_period]
                
                upper_midrange_values.append(curr_y_xmr + ((curr_upper - curr_y_xmr) / 2))
                lower_midrange_values.append(curr_y_xmr - ((curr_y_xmr - curr_lower) / 2))
            
            run_test_upper = y_values > array(upper_midrange_values)
            run_test_lower = y_values < array(lower_midrange_values)
        else:
            upper_midrange = y_xmr_func + ((npl_upper - y_xmr_func) / 2)
            lower_midrange = y_xmr_func - ((y_xmr_func - npl_lower) / 2)
            run_test_upper = y_values > upper_midrange
            run_test_lower = y_values < lower_midrange

        shape_buffer = (y_range[1] - y_range[0]) * shape_buffer_pct

        for i, el in enumerate(y_values):
            min_i = max(0, i - 3)
            max_i = i + 1
            trailing_sum_upper = numpy_sum(run_test_upper[min_i:max_i])
            trailing_sum_lower = numpy_sum(run_test_lower[min_i:max_i])
            if trailing_sum_upper >= 3:
                short_runs.append(
                    zip(
                        x_values[min_i:max_i],
                        y_values[min_i:max_i],
                        ["High"] * (max_i - min_i),
                    )
                )
            elif trailing_sum_lower >= 3:
                short_runs.append(
                    zip(
                        x_values[min_i:max_i],
                        y_values[min_i:max_i],
                        ["Low"] * (max_i - min_i),
                    )
                )
            else:
                pass

    paths = []
    for run in short_runs:
        path_build_list = []
        for i, (d, v, t) in enumerate(run):
            path_build_list.append((d, v, t))

        paths.append(path_build_list)

    # combine overlapping paths
    c_paths = combine_paths.combine_paths(paths)

    path_strings = []
    for path in c_paths:
        path_string = ""
        for i, el in enumerate(path):
            d = el[0]
            v = el[1]
            if i == 0:
                path_string += "M {} {}".format(d, v + shape_buffer)
            else:
                path_string += " L {} {}".format(d, v + shape_buffer)

        for el in path[::-1]:
            d = el[0]
            v = el[1]
            path_string += " L {} {}".format(d, v - shape_buffer)

        path_string += " Z"

        path_strings.append(path_string)

    shapes = []
    for path_string in path_strings:
        shapes.append(
            {
                "fillcolor": fill_color,
                "line": {"color": line_color, "dash": line_type, "width": line_width},
                "name": "Short Run",
                "opacity": opacity,
                "path": (path_string),
                "type": "path",
            }
        )

    return shapes, c_paths


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
    period_ranges: list = None
) -> tuple:
    """
    Identifies "long runs", defined as 8 consecutive points above or below the mid line.

    Parameters:
        fig (Figure): Passed in Figure object
        y_xmr_func (float|list): Upper moving range limit. If sloped is True, this is a list of tuples.
        sloped (bool): Use sloping approach for limit values.
        fill_color (str): Fill color of shape
        line_color (str): Line color of shape
        line_width (int): Line width of shape border
        line_type (str): Line type of shape border
        opacity (float): Opacity of shape fill
        shape_buffer_pct (float): % buffer to use for shape build. For example:
            If y-value = 100, a shape buffer of 5% would mean the lower and upper values
            of the shape for that y-value are [95, 105].

    Returns:
        list[dict]: List of dictionaries that represent a shape for each long run that will
            highlight the area of the chart containing the long run.
        list[list]: List of lists, each sublist is a path, containing a tuple that represents
            each point in the long run.
    """
    fig_data = fig.data
    x_values = [el for el in fig_data[0].x]
    y_values = fig_data[0].y

    # Convert x_values if needed
    if x_type == "date_time":
        x_values = pd.to_datetime(x_values)

    long_runs = []
    y_range = fig.layout.yaxis.range
    shape_buffer = (y_range[1] - y_range[0]) * shape_buffer_pct

    if sloped:
        y_func_values = [el[1] for el in y_xmr_func]
        run_test_upper = y_values > array(y_func_values)
        run_test_lower = y_values < array(y_func_values)

        for i, el in enumerate(y_values):
            trailing_sum_upper = numpy_sum(run_test_upper[max(0, i - 7) : i + 1])
            trailing_sum_lower = numpy_sum(run_test_lower[max(0, i - 7) : i + 1])
            if trailing_sum_upper >= 8 or trailing_sum_lower >= 8:
                long_runs.append(
                    zip(x_values[max(0, i - 7) : i + 1], y_values[max(0, i - 7) : i + 1])
                )

        paths = []
        for run in long_runs:
            path_build_list = []
            for i, (d, v) in enumerate(run):
                y_value = [el[1] for el in y_xmr_func if el[0] == d]
                path_build_list.append((d, v, "High" if v >= y_value else "Low"))

            paths.append(path_build_list)
    else:
        # Handle both single and multiple periods
        if isinstance(y_xmr_func, list):
            # Create arrays for comparison based on period ranges
            y_func_values = []
            current_period = 0
            for x in x_values:
                if period_ranges and current_period < len(period_ranges) - 1:
                    if x >= period_ranges[current_period + 1][0]:
                        current_period += 1
                y_func_values.append(y_xmr_func[current_period])
            
            run_test_upper = y_values > array(y_func_values)
            run_test_lower = y_values < array(y_func_values)
        else:
            run_test_upper = y_values > y_xmr_func
            run_test_lower = y_values < y_xmr_func

        for i, el in enumerate(y_values):
            trailing_sum_upper = numpy_sum(run_test_upper[max(0, i - 7) : i + 1])
            trailing_sum_lower = numpy_sum(run_test_lower[max(0, i - 7) : i + 1])
            if trailing_sum_upper >= 8 or trailing_sum_lower >= 8:
                long_runs.append(
                    zip(x_values[max(0, i - 7) : i + 1], y_values[max(0, i - 7) : i + 1])
                )

        paths = []
        for run in long_runs:
            path_build_list = []
            for i, (d, v) in enumerate(run):
                path_build_list.append((d, v, "High" if v >= y_xmr_func else "Low"))

            paths.append(path_build_list)

    # combine overlapping paths
    c_paths = combine_paths.combine_paths(paths)

    path_strings = []
    for path in c_paths:
        path_string = ""
        for i, el in enumerate(path):
            d = el[0]
            v = el[1]
            if i == 0:
                path_string += "M {} {}".format(d, v + shape_buffer)
            else:
                path_string += " L {} {}".format(d, v + shape_buffer)

        for el in path[::-1]:
            d = el[0]
            v = el[1]
            path_string += " L {} {}".format(d, v - shape_buffer)

        path_string += " Z"

        path_strings.append(path_string)

    shapes = []
    for path_string in path_strings:
        shapes.append(
            {
                "fillcolor": fill_color,
                "line": {"color": line_color, "dash": line_type, "width": line_width},
                "name": "Long Run",
                "opacity": opacity,
                "path": (path_string),
                "type": "path",
            }
        )

    return shapes, c_paths
