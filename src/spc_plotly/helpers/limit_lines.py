import plotly.graph_objects as go
from pandas import DataFrame
from spc_plotly.utils import endpoints, rounded_value, rounding_multiple
from numpy import array


def _limit_line_shape(
    line_color: str,
    line_type: str,
    y0: float,
    y1: float,
    yref: str,
    x0: float = 0,
    x1: float = 1,
    xref: str = "x domain",
) -> go.layout.Shape:
    return go.layout.Shape(
        {
            "line": {"color": line_color, "dash": line_type},
            "type": "line",
            "x0": x0,
            "x1": x1,
            "xref": xref,
            "y0": y0,
            "y1": y1,
            "yref": yref,
        }
    )


def _create_limit_lines(
    data: DataFrame,
    y_xmr_func: float,
    npl_upper: float,
    npl_lower: float,
    mR: float,
    mR_upper: float,
    sloped: bool,
    period_ranges: list = None,
    chart_line_color: str = "gray",
    chart_limit_color: str = "red",
    chart_midrange_color: str = "pink",
    chart_line_type: str = "dashdot",
    chart_midrange_line_type: str = "dot",
    ) -> list:
    """
    Create limit lines for X-chart and mR-Chart

    Parameters:
        data (DataFrame): All data
        y_xmr_func (float|list): Upper moving range limit. If sloped is True, this is a list of tuples.
        npl_upper (float|list): Upper process limit. If sloped is True, this is a list of tuples.
        npl_lower (float|list): Lower process limit. If sloped is True, this is a list of tuples.
        mR (float|list): Moving range mid-line.
        mr_Upper (float|list): Upper moving range limit.
        y_name (str): Y-axis title.
        sloped (bool): Use sloping approach for limit values.
        chart_line_color (str): Mid-line color
        chart_limit_color (str): Limit line color
        chart_midrange_color (str): Midrange line color (i.e., line between mid-line and limit line)
        chart_line_type (str): Mid- & limit line type
        chart_midrange_line_type (str): Midrange line type
        period_ranges (list): List of tuples containing (start, end) for each period

    Returns:
        list[go.layout.Shape]: List of shape objects representing all XmR chart lines
    """

    shapes = []
    if period_ranges:
        # Handle multiple periods
        for i, (start, end) in enumerate(period_ranges):
            period_data = data.loc[
                int(data[data == start].index[0]) : int(data[data == end].index[0])
            ]
            
            if sloped:
                # Calculate endpoints for this period
                y_endpoints = endpoints.get_line_endpoints(y_xmr_func[i], period_data)
                upper_endpoints = endpoints.get_line_endpoints(npl_upper[i], period_data)
                lower_endpoints = endpoints.get_line_endpoints(npl_lower[i], period_data)

                # Calculate midrange lines for this period
                npl_upper_mid = [
                    (mid[0], mid[1] + ((upper[1] - mid[1]) / 2))
                    for (upper, mid) in zip(npl_upper[i], y_xmr_func[i])
                ]
                npl_lower_mid = [
                    (mid[0], mid[1] - ((mid[1] - lower[1]) / 2))
                    for (lower, mid) in zip(npl_lower[i], y_xmr_func[i])
                ]

                upper_mid_endpoints = endpoints.get_line_endpoints(npl_upper_mid, period_data)
                lower_mid_endpoints = endpoints.get_line_endpoints(npl_lower_mid, period_data)

            else:
                # Calculate midrange values for this period
                upper_midrange = y_xmr_func[i] + ((npl_upper[i] - y_xmr_func[i]) / 2)
                lower_midrange = y_xmr_func[i] - ((y_xmr_func[i] - npl_lower[i]) / 2)

            mR_upper_midrange = mR[i] + ((mR_upper[i] - mR[i]) / 2)

            # Add shapes for this period
            period_shapes = [
                # Individual Values Chart
                # Average line
                _limit_line_shape(
                    line_color=chart_line_color,
                    line_type=chart_line_type,
                    y0=y_endpoints["start"]["y"] if sloped else y_xmr_func[i],
                    y1=y_endpoints["end"]["y"] if sloped else y_xmr_func[i],
                    yref="y",
                    x0=start,
                    x1=end,
                    xref="x",
                ),
                # Upper limit
                _limit_line_shape(
                    line_color=chart_limit_color,
                    line_type=chart_line_type,
                    y0=upper_endpoints["start"]["y"] if sloped else npl_upper[i],
                    y1=upper_endpoints["end"]["y"] if sloped else npl_upper[i],
                    yref="y",
                    x0=start,
                    x1=end,
                    xref="x",
                ),
                # Lower limit
                _limit_line_shape(
                    line_color=chart_limit_color,
                    line_type=chart_line_type,
                    y0=lower_endpoints["start"]["y"] if sloped else npl_lower[i],
                    y1=lower_endpoints["end"]["y"] if sloped else npl_lower[i],
                    yref="y",
                    x0=start,
                    x1=end,
                    xref="x",
                ),
                # Upper mid-range line
                _limit_line_shape(
                    line_color=chart_midrange_color,
                    line_type=chart_midrange_line_type,
                    y0=upper_mid_endpoints["start"]["y"] if sloped else upper_midrange,
                    y1=upper_mid_endpoints["end"]["y"] if sloped else upper_midrange,
                    yref="y",
                    x0=start,
                    x1=end,
                    xref="x",
                ),
                # Lower mid-range line
                _limit_line_shape(
                    line_color=chart_midrange_color,
                    line_type=chart_midrange_line_type,
                    y0=lower_mid_endpoints["start"]["y"] if sloped else lower_midrange,
                    y1=lower_mid_endpoints["end"]["y"] if sloped else lower_midrange,
                    yref="y",
                    x0=start,
                    x1=end,
                    xref="x",
                ),
                # Moving Range Chart
                # Average line
                _limit_line_shape(
                    line_color=chart_line_color,
                    line_type=chart_line_type,
                    y0=mR[i],
                    y1=mR[i],
                    yref="y2",
                    x0=start,
                    x1=end,
                    xref="x",
                ),
                # Upper limit
                _limit_line_shape(
                    line_color=chart_limit_color,
                    line_type=chart_line_type,
                    y0=mR_upper[i],
                    y1=mR_upper[i],
                    yref="y2",
                    x0=start,
                    x1=end,
                    xref="x",
                ),
                # Mid-range line
                _limit_line_shape(
                    line_color=chart_midrange_color,
                    line_type=chart_midrange_line_type,
                    y0=mR_upper_midrange,
                    y1=mR_upper_midrange,
                    yref="y2",
                    x0=start,
                    x1=end,
                    xref="x",
                ),
            ]
            shapes.extend(period_shapes)
            
            if sloped:
                # Add vertical lines for this period
                value_range = npl_upper[i][len(npl_upper[i]) - 1][1] - npl_lower[i][0][1]
                multiple = rounding_multiple.rounding_multiple(value_range)
                range_min = rounded_value.rounded_value(npl_lower[i][0][1], multiple)
                range_max = rounded_value.rounded_value(
                    npl_upper[i][len(npl_upper[i]) - 1][1], multiple, "up"
                )
                
                half_idx = period_data.shape[0] // 2
                first_half_idx = period_data.values[:half_idx].shape[0] // 2
                first_half_loc = (first_half_idx + period_data.index[0]) / data.shape[0]
                second_half_idx = period_data.values[half_idx:].shape[0] // 2
                second_half_loc = ((second_half_idx + half_idx) + period_data.index[0]) / data.shape[0]

                period_vertical_lines = [
                    {
                        "fillcolor": "gray",
                        "line": {"color": "gray", "dash": "dot", "width": 1},
                        "name": "limit sloped line",
                        "opacity": 1,
                        "path": f"M {first_half_loc} {npl_lower[i][first_half_idx][1]} L {first_half_loc} {npl_upper[i][first_half_idx][1]+((range_max-range_min)*.05)}",
                        "type": "path",
                        "xref": "paper",
                    },
                    {
                        "fillcolor": "gray",
                        "line": {"color": "gray", "dash": "dot", "width": 1},
                        "name": "limit sloped line",
                        "opacity": 1,
                        "path": f"M {second_half_loc} {npl_upper[i][second_half_idx+half_idx][1]} L {second_half_loc} {npl_lower[i][second_half_idx+half_idx][1]-((range_max-range_min)*.05)}",
                        "type": "path",
                        "xref": "paper",
                    },
                ]
                shapes.extend(period_vertical_lines)
    else:
        # Create natural limits, mid-range lines, and center lines
        if sloped:
            y_endpoints = endpoints.get_line_endpoints(y_xmr_func, data)

            upper_endpoints = endpoints.get_line_endpoints(npl_upper, data)
            lower_endpoints = endpoints.get_line_endpoints(npl_lower, data)

            npl_upper_mid = [
                (mid[0], mid[1] + ((upper[1] - mid[1]) / 2))
                for (upper, mid) in zip(npl_upper, y_xmr_func)
            ]
            npl_lower_mid = [
                (mid[0], mid[1] - ((mid[1] - lower[1]) / 2))
                for (lower, mid) in zip(npl_lower, y_xmr_func)
            ]

            upper_mid_endpoints = endpoints.get_line_endpoints(npl_upper_mid, data)
            lower_mid_endpoints = endpoints.get_line_endpoints(npl_lower_mid, data)

            half_idx = data.shape[0] // 2
            first_half_idx = data.values[:half_idx].shape[0] // 2
            first_half_loc = first_half_idx / data.shape[0]
            second_half_idx = data.values[half_idx:].shape[0] // 2
            second_half_loc = (second_half_idx + half_idx) / data.shape[0]

            value_range = npl_upper[len(npl_upper) - 1][1] - npl_lower[0][1]
            multiple = rounding_multiple.rounding_multiple(value_range)
            range_min = rounded_value.rounded_value(npl_lower[0][1], multiple)
            range_max = rounded_value.rounded_value(
                npl_upper[len(npl_upper) - 1][1], multiple, "up"
            )
            sloped_vertical_lines = [
                {
                    "fillcolor": "gray",
                    "line": {"color": "gray", "dash": "dot", "width": 1},
                    "name": "limit sloped line",
                    "opacity": 1,
                    "path": f"M {first_half_loc} {npl_lower[first_half_idx][1]} L {first_half_loc} {npl_upper[first_half_idx][1]+((range_max-range_min)*.05)}",
                    "type": "path",
                    "xref": "paper",
                },
                {
                    "fillcolor": "gray",
                    "line": {"color": "gray", "dash": "dot", "width": 1},
                    "name": "limit sloped line",
                    "opacity": 1,
                    "path": f"M {second_half_loc} {npl_upper[second_half_idx+half_idx][1]} L {second_half_loc} {npl_lower[second_half_idx+half_idx][1]-((range_max-range_min)*.05)}",
                    "type": "path",
                    "xref": "paper",
                },
            ]
        else:
            upper_midrange = y_xmr_func + ((npl_upper - y_xmr_func) / 2)
            lower_midrange = y_xmr_func - ((y_xmr_func - npl_lower) / 2)

        mR_upper_midrange = mR + ((mR_upper - mR) / 2)

        shapes = [
            # Individual Values Chart
            # Average line
            _limit_line_shape(
                line_color=chart_line_color,
                line_type=chart_line_type,
                y0=y_endpoints["start"]["y"] if sloped else y_xmr_func,
                y1=y_endpoints["end"]["y"] if sloped else y_xmr_func,
                yref="y",
            ),
            # Limit Lines
            _limit_line_shape(
                line_color=chart_limit_color,
                line_type=chart_line_type,
                y0=upper_endpoints["start"]["y"] if sloped else npl_upper,
                y1=upper_endpoints["end"]["y"] if sloped else npl_upper,
                yref="y",
            ),
            _limit_line_shape(
                line_color=chart_limit_color,
                line_type=chart_line_type,
                y0=lower_endpoints["start"]["y"] if sloped else npl_lower,
                y1=lower_endpoints["end"]["y"] if sloped else npl_lower,
                yref="y",
            ),
            # Mid-Range Lines
            _limit_line_shape(
                line_color=chart_midrange_color,
                line_type=chart_midrange_line_type,
                y0=upper_mid_endpoints["start"]["y"] if sloped else upper_midrange,
                y1=upper_mid_endpoints["end"]["y"] if sloped else upper_midrange,
                yref="y",
            ),
            _limit_line_shape(
                line_color=chart_midrange_color,
                line_type=chart_midrange_line_type,
                y0=lower_mid_endpoints["start"]["y"] if sloped else lower_midrange,
                y1=lower_mid_endpoints["end"]["y"] if sloped else lower_midrange,
                yref="y",
            ),
            # Moving Range Chart
            # Average line
            _limit_line_shape(
                line_color=chart_line_color,
                line_type=chart_line_type,
                y0=mR,
                y1=mR,
                yref="y2",
            ),
            # Limit Lines
            _limit_line_shape(
                line_color=chart_limit_color,
                line_type=chart_line_type,
                y0=mR_upper,
                y1=mR_upper,
                yref="y2",
            ),
            # Mid-Range Lines
            _limit_line_shape(
                line_color=chart_midrange_color,
                line_type=chart_midrange_line_type,
                y0=mR_upper_midrange,
                y1=mR_upper_midrange,
                yref="y2",
            ),
        ]

        if sloped:
            shapes.extend(sloped_vertical_lines)

    return shapes
