from pandas import Series, DataFrame
import numpy as np
from plotly.graph_objects import Figure, Scatter
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def _create_base_traces(
    x_Ser: Series,
    y_Ser: Series,
    mR_data: Series,
    x_type: str,
    middle: list,
    middle_mr: list,
    middle_type: str,
    chart_line_color: str = "black",
    chart_marker_color: str = "black",
    chart_line_width: int = 1,
    chart_marker_size: int = 6,
) -> list:
    """Create base traces for XmR chart"""
    mR_data_new = mR_data[0:len(y_Ser)]

    df = DataFrame({   #this could be done without the df but was a PITA
        'x_ser': list(x_Ser),
        'y_ser': list(y_Ser),
        'mR_data': list(mR_data_new),
        'middle': middle,
        'middle_mr': middle_mr
    })
    
    # Handle hover template based on x-axis type
    if x_type == "date_time":
        hovertemplate = (
            "<b>Date:</b> %{x|%Y-%m-%d}<br>"
            "<b>Value:</b> %{y:,.2f}<br>"
            f"<b>{middle_type.capitalize()}:</b>"
            "%{customdata[0]:,.2f}<br>"
            "<extra></extra>"
        )
    elif x_type == "numeric":
        hovertemplate = (
            "<b>Position:</b> %{x}<br>"
            "<b>Value:</b> %{y:,.2f}<br>"
            f"<b>{middle_type.capitalize()}:</b>"
            "%{customdata[0]:,.2f}<br>"
            "<extra></extra>"
        )
    else:  # categorical
        hovertemplate = (
            "<b>Category:</b> %{x}<br>"
            "<b>Value:</b> %{y:,.2f}<br>"
            f"<b>{middle_type.capitalize()}:</b> "
            "%{customdata[0]:,.2f}<br>"
            "<extra></extra>"
    )

    traces = [
        go.Scatter(
            x=df['x_ser'],
            y=df['y_ser'],
            mode="lines+markers",
            name=y_Ser.name,
            line={"color": chart_line_color, "width": chart_line_width},
            marker={"color": chart_marker_color, "size": chart_marker_size},
            hovertemplate=hovertemplate,
            customdata=np.stack( (df['middle'], df['middle_mr']), axis=-1)
        ),
        go.Scatter(
            x=df['x_ser'],
            y=df['mR_data'],
            mode="lines+markers",
            name="Moving Range",
            line={"color": chart_line_color, "width": chart_line_width},
            marker={"color": chart_marker_color, "size": chart_marker_size},
            xaxis="x2",
            yaxis="y2",
            hovertemplate=hovertemplate.replace("Value", "Range"),
            customdata=np.stack( (df['middle_mr'], df['middle']), axis=-1)
        ),
    ]
    return traces
