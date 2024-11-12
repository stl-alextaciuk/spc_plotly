from pandas import DataFrame
from spc_plotly.utils import rounded_value, rounding_multiple


def _limit_line_annotation(
    font: dict,
    text: str,
    x: float,
    xanchor: str,
    xref: str,
    y: float,
    yanchor: str,
    yref: str,
    showarrow: bool = False,
):
    """
    Annotation formatter. For documentation, see plotly's official docs

    Parameters:
        font (dict):
        text (str):
        x (float):
        xanchor (str):
        xref (str):
        y (float):
        yanchor (str):
        yref (str):
        showarrow (bool):

    Returns:
        dict: Axis formatting
    """
    return {
        "font": font,
        "text": text,
        "x": x,
        "xanchor": xanchor,
        "xref": xref,
        "y": y,
        "yanchor": yanchor,
        "yref": yref,
        "showarrow": showarrow,
    }


def _create_limit_line_annotations(
    data: DataFrame,
    chart_title: str,
    y_xmr_func,
    mR_upper,
    mR_xmr_func,
    npl_upper,
    npl_lower,
    y_name: str,
    sloped: bool,
):
    """
    Annotation formatter for limit lines and chart title.

    Parameters:
        data (DataFrame): All data
        chart_title (str): Chart title
        y_xmr_func (float|list): Natural process limit mid-line.
            If sloped is True, this is a list of tuples.
        mR_upper (float|list): Upper moving range limit.
        mR_xmr_func (float|list): Moving range mid-line.
        npl_upper (float|list): Upper process limit. If sloped is True, this is a list of tuples.
        npl_lower (float|list): Lower process limit. If sloped is True, this is a list of tuples.
        y_name (str): Y-axis title.
        sloped (bool): Use sloping approach for limit values.

    Returns:
        list: List of annotation dictionaries
    """
    other_font = {"size": 10}
    other_x = 0.01
    other_xanchor = "left"
    other_xref = "paper"

    # Create title annotation
    annotations = [
        _limit_line_annotation(
            font={"size": 16},
            text=chart_title,
            x=0.5,
            xanchor="center",
            xref="paper",
            y=1.1,
            yanchor="top",
            yref="paper",
        )
    ]

    if sloped:
        # For sloped lines, calculate annotations for middle points
        half_idx = data.shape[0] // 2
        first_half_idx = data.values[:half_idx].shape[0] // 2
        first_half_loc = first_half_idx / data.shape[0]
        second_half_idx = data.values[half_idx:].shape[0] // 2
        second_half_loc = (second_half_idx + half_idx) / data.shape[0]

        # Calculate value range for spacing
        value_range = npl_upper[len(npl_upper) - 1][1] - npl_lower[0][1]

        x_annotations = [
            _limit_line_annotation(
                font=other_font,
                text=f"<b>{float(y_xmr_func[first_half_idx][1]):.2f} "
                + "\u00B1"
                + f" {float(mR_xmr_func):.2f}</b>",
                x=first_half_loc,
                xanchor="center",
                xref="paper",
                y=npl_upper[first_half_idx][1] + (value_range * 0.1),
                yanchor="auto",
                yref="y",
            ),
            _limit_line_annotation(
                font=other_font,
                text=f"<b>{float(y_xmr_func[second_half_idx+half_idx][1]):.2f} "
                + "\u00B1"
                + f" {float(mR_xmr_func):.2f}</b>",
                x=second_half_loc,
                xanchor="center",
                xref="paper",
                y=npl_lower[half_idx + second_half_idx][1] - (value_range * 0.1),
                yanchor="auto",
                yref="y",
            ),
        ]
    else:
        # For non-sloped lines
        if isinstance(npl_upper, list):
            # Multiple periods - use first period for annotations
            value_range = npl_upper[0] - npl_lower[0]
            x_annotations = [
                _limit_line_annotation(
                    font=other_font,
                    text=f"<b>{y_name} Upper Limit = {float(npl_upper[0]):.3f}</b>",
                    x=other_x,
                    xanchor=other_xanchor,
                    xref=other_xref,
                    y=npl_upper[0] + (value_range * 0.03),
                    yanchor="auto",
                    yref="y",
                ),
                _limit_line_annotation(
                    font=other_font,
                    text=f"<b>{y_name} Lower Limit = {float(npl_lower[0]):.3f}</b>",
                    x=other_x,
                    xanchor=other_xanchor,
                    xref=other_xref,
                    y=npl_lower[0] - (value_range * 0.03),
                    yanchor="auto",
                    yref="y",
                ),
            ]
        else:
            # Single period
            value_range = npl_upper - npl_lower
            x_annotations = [
                _limit_line_annotation(
                    font=other_font,
                    text=f"<b>{y_name} Upper Limit = {float(npl_upper):.3f}</b>",
                    x=other_x,
                    xanchor=other_xanchor,
                    xref=other_xref,
                    y=npl_upper + (value_range * 0.03),
                    yanchor="auto",
                    yref="y",
                ),
                _limit_line_annotation(
                    font=other_font,
                    text=f"<b>{y_name} Lower Limit = {float(npl_lower):.3f}</b>",
                    x=other_x,
                    xanchor=other_xanchor,
                    xref=other_xref,
                    y=npl_lower - (value_range * 0.03),
                    yanchor="auto",
                    yref="y",
                ),
            ]

    # Add mR annotations
    if isinstance(mR_upper, list):
        annotations.append(
            _limit_line_annotation(
                font=other_font,
                text=f"<b>mR Upper Limit = {float(mR_upper[0]):.3f}</b>",
                x=other_x,
                xanchor=other_xanchor,
                xref=other_xref,
                y=mR_upper[0] + (mR_upper[0] * 0.05),
                yanchor="auto",
                yref="y2",
            )
        )
    else:
        annotations.append(
            _limit_line_annotation(
                font=other_font,
                text=f"<b>mR Upper Limit = {float(mR_upper):.3f}</b>",
                x=other_x,
                xanchor=other_xanchor,
                xref=other_xref,
                y=mR_upper + (mR_upper * 0.05),
                yanchor="auto",
                yref="y2",
            )
        )

    annotations.extend(x_annotations)
    return annotations
