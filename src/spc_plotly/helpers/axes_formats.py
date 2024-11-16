from pandas import Series
import plotly.graph_objects as go
from spc_plotly.utils import rounded_value, rounding_multiple

def _format_xaxis(anchor: str, x_type: str, showticklabels=True) -> dict:
    """
    X-axis formatter

    Parameters:
        anchor (str): Axis to anchor on
        showticklabels (bool): Show axis ticklabels
        x_type (str): Type of data you're charting

    Returns:
        dict: Axis formatting
    """
    base_format = {
        "anchor": anchor,
        "domain": [0.0, 1.0],
        "automargin": True,
        "showticklabels": showticklabels,
        "tickformat": "%b\n%Y",
        "ticklabelmode": "period",
        "tickangle": 0,
        "showspikes": True,
        "spikemode": "across+toaxis",
        "spikesnap": "cursor",
        "showline": True,
        "showgrid": True,
        "gridcolor": "lightgray",
        "zeroline": False,
    }
    
    if x_type == "date_time":
        base_format.update({
            "type": "date",
            "tickformat": "%Y-%m-%d",
            "dtick": "M1", #fix this so it's not auto-default

        })
    elif x_type == "numeric":
        base_format.update({
            "type": "linear",
            "tickformat": "d"
        })
    else:  # categorical
        base_format.update({
            "type": "category",
            "tickangle": 45,
            "automargin": True  # Ensure there's enough margin for rotated labels
        })
    
    return base_format

def _format_yaxis(
    anchor: str, title: str, domain: list, range: list, tickformat: str, dtick: float
):
    """
    Formats the y-axis configuration for a Plotly chart with specified parameters.

    Parameters:
        anchor (str): The x-axis to anchor this y-axis to (e.g., 'x', 'x2')
        title (str): The title text to display for the y-axis
        domain (list): A list of two floats defining the axis domain range [start, end]
            where values must be between 0 and 1
        range (list): A list of two values defining the axis range [min, max]
        tickformat (str): The format string for tick labels (e.g., "0" for integers)
        dtick (float): The interval between ticks on the axis

    Returns:
        dict: A dictionary containing all the formatting specifications for the y-axis
            that can be used in a Plotly figure layout
    """
    return {
        "anchor": anchor,
        "title": {"text": title},
        "domain": domain,
        "range": range,
        "tickformat": tickformat,
        "dtick": dtick,
    }


def _format_XmR_axes(
    npl_upper: float | list,
    npl_lower: float | list,
    mR_upper: float | list,
    y_Ser: Series,
    mR_data: Series,
    x_type: str,
    sloped: bool,
    y_axis_dtick: float,
) -> go.Figure:
    """
    Apply axes formats

    Parameters:
        npl_upper (float|list): Upper process limit. If sloped is True, this is a list of tuples.
        npl_lower (float|list): Lower process limit. If sloped is True, this is a list of tuples.
        mR_upper (float): Upper moving range limit.
        y_Ser (Series): Series of y-values
        mR_data (Series): Series of moving range values
        sloped (bool): Use sloping approach for limit values.
        y_axis_dtick (float): Float for y-axis interval, defaults to auto-selection

    Returns:
        dict: Axis formatting
    """
    if sloped:
        value_range = npl_upper[len(npl_upper) - 1][1] - npl_lower[0][1]
        dtick = y_axis_dtick if y_axis_dtick else rounding_multiple.rounding_multiple(value_range)
        min_range = rounded_value.rounded_value(npl_lower[0][1], dtick)
        max_range = rounded_value.rounded_value(
            npl_upper[len(npl_upper) - 1][1], dtick, "up"
        )
    else:
        # Handle both single values and lists
        if isinstance(npl_upper, list):
            # For multiple periods, use the overall min/max
            value_range = max(npl_upper) - min(npl_lower)
            dtick = y_axis_dtick if y_axis_dtick else rounding_multiple.rounding_multiple(value_range)
            min_range = min(
                rounded_value.rounded_value(y_Ser.min(), dtick, "down"),
                rounded_value.rounded_value(min(npl_lower) - (value_range * 0.1), dtick, "down"),
            )
            max_range = max(
                rounded_value.rounded_value(y_Ser.max(), dtick, "up"),
                rounded_value.rounded_value(max(npl_upper) + (value_range * 0.1), dtick, "up"),
            )
        else:
            # Original single-period logic
            value_range = npl_upper - npl_lower
            dtick = y_axis_dtick if y_axis_dtick else rounding_multiple.rounding_multiple(value_range)
            min_range = min(
                rounded_value.rounded_value(y_Ser.min(), dtick, "down"),
                rounded_value.rounded_value(npl_lower - (value_range * 0.1), dtick, "down"),
            )
            max_range = max(
                rounded_value.rounded_value(y_Ser.max(), dtick, "up"),
                rounded_value.rounded_value(npl_upper + (value_range * 0.1), dtick, "up"),
            )

    yaxis_values = _format_yaxis(
        anchor="x",
        title=y_Ser.name,
        domain=[0.4, 1.0],
        range=[min_range, max_range],
        tickformat="0",
        dtick=dtick,
    )

    xaxis_values = _format_xaxis(
        anchor="y", 
        x_type=x_type, 
    )
    dtick = y_axis_dtick if y_axis_dtick else rounding_multiple.rounding_multiple(mR_upper)
    mR_value_range = mR_upper  # Since mR chart always starts at 0
    if isinstance(mR_value_range, list):
        mR_dtick = y_axis_dtick if y_axis_dtick else rounding_multiple.rounding_multiple(max(mR_value_range)*2)
    else:
        mR_dtick = y_axis_dtick if y_axis_dtick else rounding_multiple.rounding_multiple(mR_value_range*2)
    max_range = max(
        rounded_value.rounded_value(mR_upper, mR_dtick, "up"),
        rounded_value.rounded_value(mR_data.max() + (mR_data.max() * 0.1), mR_dtick, "up"),
    )

    yaxis_mR = _format_yaxis(
        anchor="x2",
        title="Moving Range",
        domain=[0.0, 0.3],
        range=[0, max_range],
        tickformat="0",
        dtick=mR_dtick,  # Use the new dtick calculation
    )

    return {
        "x_values": xaxis_values,
        # "x_mR": xaxis_mR,
        "y_values": yaxis_values,
        "y_mR": yaxis_mR,
    }
