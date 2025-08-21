from pandas import DataFrame, Series, to_datetime
import pandas as pd
from numpy import abs
import numpy as np
from spc_plotly.helpers import (
    axes_formats,
    base_traces,
    limit_lines,
    annotations,
    signals,
    menus,
)
from spc_plotly.helpers.signals import _get_run_identifiers_for_tooltips
from tests import test_xmr
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots
from spc_plotly.utils import calc_xmr_func, validate_sequence


XmR_constants = {
    "mean": {"mR_Upper": 3.268, "npl_Constant": 2.660},
    "median": {"mR_Upper": 3.865, "npl_Constant": 3.145},
}

date_parts = {
    "year": "%Y",
    "month": "%Y-%m",
    "day": "%Y-%m-%d",
    "hour": "%Y-%m-%d %H",
    "minute": "%Y-%m-%d %H:%M",
    "custom": None,
}

x_type_options = {
    "date_time": date_parts,
    "numeric": None,
    "categorical": None
}


class XmR:
    """
    Create an XmR (Individuals and Moving Range) chart with SPC signals, period support,
    and rich tooltips/menus.

    Overview:
        - Builds an Individuals chart (X) and a Moving Range chart (mR)
        - Computes limits and center lines using mean or median
        - Supports period segmentation via `period_breaks` and a fixed `period_end` for the final period
        - Detects and visualizes SPC signals:
            - "Anomalies" (points beyond natural process limits)
            - "Long runs" (8+ consecutive points on the same side of the center)
            - "Short runs" (3 of 4 consecutive points closer to a limit than the center; min highlighted block is 4)
        - Provides interactive menu buttons to toggle anomalies, long runs, short runs, or none/all
        - Enhances tooltips with boolean flags for anomaly, short run, and long run

    Notes:
        - Period logic:
            - `period_breaks` splits the data into periods; limits are recalculated per period
            - `period_end` (if provided) applies only to the final period and restricts the data used to compute its limits; all points to the right reuse those limits
        - Date/time handling:
            - Internally, date-time values are coerced to timezone-naive midnights for stable alignment of points, shapes, and breaks
            - Shape path timestamps are encoded in UTC to match Plotly's rendering
        - Short/long runs visualization is built from the same boolean flags used in tooltips, ensuring visual and tooltip consistency

    Attributes:
        data (DataFrame): Source data used for the chart.
        y_ser_name (str): Column containing y-axis values.
        x_ser_name (str): Column (or index) containing ordered x-axis values (date-time, numeric, or categorical).
        period_breaks (list|None): Points at which limits are recalculated using only data within each period.
        period_end (str|int|None): Furthest x value used to compute the final period's limits; later points reuse those limits.
        x_type (str): One of {"date_time", "numeric", "categorical"}.
        date_part_resolution (str): If `x_type` is "date_time", one of {"year","month","day","hour","minute","custom"}.
        custom_date_part (str): If resolution is "custom", the d3 date format string to use for axis labels.
        x_begin (str|int|None): Optional left bound for limits calculations.
        x_cutoff (str|int|None): Optional right bound for limits calculations.
        title (str|None): Custom chart title.
        sloped (bool): Use sloping center/limits (kept for compatibility; typically disabled in this build).
        xmr_function (str): "mean" or "median" used for center and limits.
        chart_height (int|None): Chart height override.
        y_axis_dtick (float|None): y-axis tick spacing override.
    """

    def __init__(
        self,
        data: DataFrame,
        y_ser_name: str,
        x_ser_name: str,
        period_breaks: list[str] = None,
        period_end: str | int = None,
        x_type: str = "date_time",
        date_part_resolution: str = "month",
        custom_date_part: str = "",
        x_begin: str | int = None,
        x_cutoff: str | int = None,
        title: str = None,
        sloped: bool = False,
        xmr_function: str = "mean",
        chart_height: int = None,
        y_axis_dtick: float = None,
    ) -> None:
        """
        Initialize an XmR chart with periods, optional final-period restriction, SPC signals,
        and enhanced tooltips.

        Parameters:
            data (DataFrame): Input data for the chart.
            y_ser_name (str): Column containing y-axis values.
            x_ser_name (str): Column or index containing ordered x values (date-time, numeric, categorical).
            period_breaks (list[str]|None): Optional list defining period boundaries. Limits (center and natural limits)
                are recalculated per period using only the data within that period.
            period_end (str|int|None): Optional rightmost x value used to compute the final period's limits. All points
                to the right of this value are evaluated against the limits established up to `period_end`.
                Accepts scalar, list, numpy array, or pandas Series; a non-empty sequence is resolved to its last value.
            x_type (str): One of {"date_time", "numeric", "categorical"}. Controls axis formatting and internal
                handling (date-time coerced to tz-naive midnights for alignment).
            date_part_resolution (str): If `x_type=="date_time"`, one of {"year","month","day","hour","minute","custom"}.
            custom_date_part (str): If `date_part_resolution=="custom"`, a d3 date format string for the x-axis.
            x_begin (str|int|None): Optional left bound for limit computations; defaults to the minimum x.
            x_cutoff (str|int|None): Optional right bound for limit computations; defaults to the maximum x.
            title (str|None): Optional chart title, otherwise derived from `y_ser_name` and x label.
            sloped (bool): Whether to compute sloped center/limits (kept for compatibility; typically disabled).
            xmr_function (str): "mean" or "median" for center and limits calculations.
            chart_height (int|None): Chart height.
            y_axis_dtick (float|None): y-axis tick spacing.

        Returns:
            None

        Behavior:
            - Period segmentation: When `period_breaks` is provided, data is split and limits are computed per period.
            - Final-period restriction: When `period_end` is provided, only points up to `period_end` in the final period
              are used to compute that period's limits; all points to the right reuse those limits.
            - Signals and visuals:
                - Anomalies: points beyond natural limits (upper/lower NPL)
                - Long runs: 8+ consecutive points on the same side of the center
                - Short runs: 3 of 4 points closer to the limit than the center (first point must be closer to a limit);
                  highlighted blocks have a minimum span of 4 points
              Tooltip booleans (anomaly/short/long) drive the shape highlights for visual consistency.
        """

        self.data = data
        self.xmr_function = xmr_function.lower()
        self.sloped = sloped
        self.x_type = x_type.lower()
        self.period_breaks = period_breaks
        # Normalize period_end to a single scalar value (handles list/tuple/ndarray/Series)
        if period_end is not None:
            pe = period_end
            try:
                import numpy as _np  # type: ignore
            except Exception:
                _np = None
            if isinstance(pe, (list, tuple)):
                pe = pe[-1] if len(pe) > 0 else None
            elif _np is not None and isinstance(pe, _np.ndarray):
                pe = pe[-1] if pe.size > 0 else None
            elif isinstance(pe, pd.Series):
                pe = pe.iloc[-1] if pe.shape[0] > 0 else None
            self.period_end = pe
        else:
            self.period_end = period_end
        if period_breaks:
            if x_type == "date_time":  # Convert breaks to datetime if using date_time x_type
                self.period_breaks = [pd.to_datetime(d) for d in period_breaks]
            self.period_breaks.sort()

            # Validate period_end is after the last period_break
            if self.period_end is not None:
                if x_type == "date_time":
                    period_end_datetime = pd.to_datetime(self.period_end)
                    if period_end_datetime <= self.period_breaks[-1]:
                        raise ValueError(f"period_end ({self.period_end}) must be after the last period_break ({self.period_breaks[-1]})")
                else:
                    if self.period_end <= self.period_breaks[-1]:
                        raise ValueError(f"period_end ({self.period_end}) must be after the last period_break ({self.period_breaks[-1]})")
        else:
            self.period_ranges = None
            self.period_ranges_index = None
        
        # Validate and process x-axis type
        test_xmr.test_x_type(self.x_type, x_type_options)
        
        # Handle date_time specific parameters
        if self.x_type == "date_time":
            self.date_part_resolution = date_part_resolution.lower()
            if self.date_part_resolution == "custom":
                self.custom_date_part = custom_date_part
            else:
                self.custom_date_part = date_parts.get(self.date_part_resolution, None)
            test_xmr.test_date_resolution(self.date_part_resolution, date_parts)

        # Validate sequence and sort if needed
        is_sorted, has_gaps = validate_sequence.validate_sequence(
            data, x_ser_name, self.x_type
        )
        if not is_sorted:
            self.data = self.data.sort_values(by=x_ser_name)

        test_xmr.test_y_ser_name_val(y_ser_name, self.data)
        self._y_ser_name = y_ser_name
        self._y_Ser = data[self._y_ser_name]

        test_xmr.test_x_ser_name_val(x_ser_name, self.data)
        self._x_ser_name = x_ser_name

        self._x_Ser = (
            data[self._x_ser_name] if self._x_ser_name in data.columns else data.index
        )

        # Convert dates if x_type is date_time
        if self.x_type == "date_time":
            test_xmr.test_x_ser_is_date(self._x_Ser)
            # Coerce to tz-naive midnight so everything (points, breaks, shapes) aligns
            s = to_datetime(self._x_Ser)
            try:
                # If tz-aware, drop tz
                s = s.dt.tz_convert('UTC').dt.tz_localize(None)
            except Exception:
                try:
                    s = s.dt.tz_localize(None)
                except Exception:
                    pass
            self._x_Ser_dt = s.dt.normalize()
            self._x_Ser = self._x_Ser_dt
        else:
            self._x_Ser_dt = self._x_Ser

        test_xmr.test_cutoff_val(x_cutoff, self._x_Ser)
        self.x_cutoff = self._x_Ser.max() if x_cutoff is None else x_cutoff

        test_xmr.test_begin_val(x_begin, self._x_Ser)
        self.x_begin = self._x_Ser.min() if x_begin is None else x_begin

        self._title = (
            f"{y_ser_name} XmR Chart by {self._x_Ser.name}" if title is None else title
        )

        # Set constant values for mean or median
        self.mR_Upper_Constant = XmR_constants.get(xmr_function).get("mR_Upper")
        self.npl_Constant = XmR_constants.get(xmr_function).get("npl_Constant")

        # Calculate limit values
        (
            self.data_for_limits,
            self.mR_data,
            self.mR_limit_values,
            self.npl_limit_values,
        ) = self._limits()

        # get the indexes of the period ranges
        if self.period_ranges:
            self.period_ranges_index = []
            for start, end in self.period_ranges:
                self.period_ranges_index.append((int(self._x_Ser[self._x_Ser == start].index[0]), int(self._x_Ser[self._x_Ser == end].index[0])))

        # Add selected function to mR and npl dictionaries for reference
        self.mR_limit_values["xmr_func"] = self.xmr_function
        self.npl_limit_values["xmr_func"] = self.xmr_function

        self._height = chart_height
        self.y_axis_dtick = y_axis_dtick
        self.xmr_chart, self.signals = self._XmR_chart()

    def _limits(self) -> tuple[DataFrame, Series, dict, dict]:
        """
        Calculate center and limits for the Individuals and Moving Range charts.

        Behavior:
            - If no `period_breaks` are provided, computes a single set of limits over the selected window.
              If `period_end` is provided, restricts the data used to compute limits up to (and including) `period_end`
              but returns full data for plotting/evaluation.
            - If `period_breaks` are provided, splits into periods and computes limits per period. If `period_end` is
              provided, it applies only to the final period as described above.

        Returns:
            tuple:
                - DataFrame: Data used for limits calculation (full data or per-period concatenation)
                - Series: Moving range series aligned to the Individuals series
                - dict: mR limits dict with keys {"mR_xmr_func", "mR_upper_limit"}
                - dict: NPL limits dict with keys {"y_xmr_func", "npl_upper_limit", "npl_lower_limit"}
        """
        if self.period_breaks == None:
            # Handle period_end for single period case
            if self.period_end is not None:
                # Find the index for period_end
                if self.x_type == "date_time":
                    period_end_datetime = pd.to_datetime(self.period_end)
                    end_mask = self._x_Ser_dt <= period_end_datetime
                else:
                    end_mask = self._x_Ser <= self.period_end
                
                if end_mask.any():
                    end_idx = end_mask[end_mask].index[-1]
                    limit_data = self.data.iloc[:end_idx + 1]
                    # Calculate limits using limited data but return full dataset
                    return self._calculate_period_limits_with_restriction(self.data, limit_data)
            
            return self._calculate_period_limits(self.data)
            
#test for duplicaters too 
        test_xmr.test_sorted(self.period_breaks)
        
        # Calculate period ranges
        self.period_ranges = []
        start_date = self.x_begin or self._x_Ser.min()
        end_date = self.x_cutoff or self._x_Ser.max()
        
        # probably a better way to do this but basically creates the period lists if/else they are are dates
        if self.x_type == 'date_time':
            # Build naive (tz-unaware) midnight boundaries so they match plotted points
            def _to_local_naive_midnight(x):
                return pd.to_datetime(x).normalize()

            self.period_ranges.append((_to_local_naive_midnight(start_date), _to_local_naive_midnight(self.period_breaks[0])))
            for i in range(len(self.period_breaks) - 1):
                self.period_ranges.append((_to_local_naive_midnight(self.period_breaks[i]), _to_local_naive_midnight(self.period_breaks[i + 1])))
            self.period_ranges.append((_to_local_naive_midnight(self.period_breaks[-1]), _to_local_naive_midnight(end_date)))

            # Keep internal series as naive datetimes for matching
            self._x_Ser = pd.to_datetime(self._x_Ser)
        else:
            self.period_ranges.append((start_date, self.period_breaks[0]))
            for i in range(len(self.period_breaks) - 1):
                self.period_ranges.append((self.period_breaks[i], self.period_breaks[i + 1]))
            self.period_ranges.append((self.period_breaks[-1], end_date))
        
        # Calculate limits for each period
        period_results = []
        for i, (start, end) in enumerate(self.period_ranges):
            # Robust index lookup per x_type
            if self.x_type == 'date_time':
                # Use tz-naive normalized dates to match period_ranges built above
                x_ser_indexable = pd.to_datetime(self._x_Ser).dt.normalize()
                start_pos = int(x_ser_indexable.searchsorted(start, side='left'))
                end_pos = int(x_ser_indexable.searchsorted(end, side='right')) - 1
            else:
                # Numeric/categorical: use Series values directly
                x_ser_indexable = pd.Series(self._x_Ser)
                # Prefer exact match if present; otherwise fall back to positional search
                eq_start = (x_ser_indexable == start).to_numpy().nonzero()[0]
                eq_end = (x_ser_indexable == end).to_numpy().nonzero()[0]
                if eq_start.size:
                    start_pos = int(eq_start[0])
                else:
                    start_pos = int(pd.Index(x_ser_indexable).searchsorted(start, side='left'))
                if eq_end.size:
                    end_pos = int(eq_end[-1])
                else:
                    end_pos = int(pd.Index(x_ser_indexable).searchsorted(end, side='right')) - 1
            period_data = self.data.iloc[start_pos : end_pos + 1]
            
            # Check if this is the final period and period_end is specified
            is_final_period = (i == len(self.period_ranges) - 1)
            if is_final_period and self.period_end is not None:
                # Apply period_end to the final period
                if self.x_type == "date_time":
                    period_end_datetime = pd.to_datetime(self.period_end).normalize()
                    # Find the end index within this period
                    period_end_series = pd.to_datetime(period_data[self._x_ser_name]).dt.normalize()
                    period_end_mask = period_end_series <= period_end_datetime
                else:
                    period_end_mask = period_data[self._x_ser_name] <= self.period_end
                
                if period_end_mask.any():
                    # Get the last index that satisfies the period_end condition
                    end_idx = period_end_mask[period_end_mask].index[-1]
                    # Calculate relative index within the period
                    period_start_idx = period_data.index[0]
                    relative_end_idx = end_idx - period_start_idx + 1
                    limit_data = period_data.iloc[:relative_end_idx]
                    
                    # Use period_end restriction for final period
                    period_results.append(self._calculate_period_limits_with_restriction(period_data, limit_data))
                else:
                    period_results.append(self._calculate_period_limits(period_data))
            else:
                period_results.append(self._calculate_period_limits(period_data))
        
        # Combine results
        combined_data = pd.concat([r[0] for r in period_results])
        combined_mR = pd.concat([r[1] for r in period_results])
        
        # Create lists of limit values for each period
        mR_limits = {
            "mR_xmr_func": [r[2]["mR_xmr_func"] for r in period_results],
            "mR_upper_limit": [r[2]["mR_upper_limit"] for r in period_results]
        }
        
        npl_limits = {
            "y_xmr_func": [r[3]["y_xmr_func"] for r in period_results],
            "npl_upper_limit": [r[3]["npl_upper_limit"] for r in period_results],
            "npl_lower_limit": [r[3]["npl_lower_limit"] for r in period_results]
        }
        
        return combined_data, combined_mR, mR_limits, npl_limits

    def _calculate_period_limits(self, period_data: DataFrame) -> tuple:
        """
        Calculate center and limits for a single period using the configured XmR function.

        Parameters:
            period_data (DataFrame): Slice of the overall dataset representing one period.

        Returns:
            tuple:
                - DataFrame: The input `period_data`
                - Series: Moving range values computed from `period_data`
                - dict: mR limits {"mR_xmr_func", "mR_upper_limit"}
                - dict: NPL limits {"y_xmr_func", "npl_upper_limit", "npl_lower_limit"}
        """
        # Use period_data instead of self.data
        data_for_limits = period_data

        # Calculate moving range data for the limits
        mR_data_for_limits = abs(
            data_for_limits[self._y_ser_name]
            - data_for_limits[self._y_ser_name].shift(1)
        )

        # Calculate mean/median values
        mR_xmr_func = calc_xmr_func.calc_xmr_func(mR_data_for_limits, self.xmr_function)
        y_xmr_func = calc_xmr_func.calc_xmr_func(
            data_for_limits[self._y_ser_name], self.xmr_function
        )

        mR_upper = mR_xmr_func * self.mR_Upper_Constant

        if self.sloped:
            # According to "Understanding Variation: The Key to Managing Chaos",
            #   we derive the slope of the mean/median line by getting the mean/median
            #   of the first half and second half of the data. We then solve for the
            #   y-intercept (b) with the slope
            half_idx = data_for_limits.shape[0] // 2
            first_half_idx = data_for_limits.values[:half_idx].shape[0] // 2
            second_half_idx = data_for_limits.values[half_idx:].shape[0] // 2

            x_delta = (half_idx + second_half_idx) - first_half_idx
            y_delta = calc_xmr_func.calc_xmr_func(
                data_for_limits[self._y_ser_name].values[half_idx:], self.xmr_function
            ) - calc_xmr_func.calc_xmr_func(
                data_for_limits[self._y_ser_name].values[:half_idx], self.xmr_function
            )
            m = y_delta / x_delta
            b = calc_xmr_func.calc_xmr_func(
                data_for_limits[self._y_ser_name].values[:half_idx], self.xmr_function
            ) - (m * (first_half_idx))

            # Create list representing upper, mid, and lower sloped paths
            sloped_path = [(i, (((i + 1) * m) + b)) for i in range(self.data.shape[0])]
            lower_limit_sloped_path = [
                (i[0], i[1] - (mR_xmr_func * self.mR_Upper_Constant))
                for i in sloped_path
            ]
            upper_limit_sloped_path = [
                (i[0], i[1] + (mR_xmr_func * self.mR_Upper_Constant))
                for i in sloped_path
            ]

            return (
                data_for_limits,
                mR_data_for_limits,  # Use the calculated moving range for the limits
                {"mR_xmr_func": mR_xmr_func, "mR_upper_limit": mR_upper},
                {
                    "y_xmr_func": sloped_path,
                    "npl_upper_limit": upper_limit_sloped_path,
                    "npl_lower_limit": lower_limit_sloped_path,
                },
            )
        else:
            npl_upper = y_xmr_func + (self.npl_Constant * mR_xmr_func)
            npl_lower = max(y_xmr_func - (self.npl_Constant * mR_xmr_func), 0)

            return (
                data_for_limits,
                mR_data_for_limits,  # Use the calculated moving range for the limits
                {"mR_xmr_func": mR_xmr_func, "mR_upper_limit": mR_upper},
                {
                    "y_xmr_func": y_xmr_func,
                    "npl_upper_limit": npl_upper,
                    "npl_lower_limit": npl_lower,
                },
            )

    def _calculate_period_limits_with_restriction(self, period_data: DataFrame, limit_data: DataFrame) -> tuple:
        """
        Calculate period limits using a restricted subset for calculations but returning full period data.

        Parameters:
            period_data (DataFrame): Full period data to be plotted and evaluated.
            limit_data (DataFrame): Subset of `period_data` used to compute center and limits (e.g., up to `period_end`).

        Returns:
            tuple:
                - DataFrame: `period_data` (full period) for plotting/evaluation
                - Series: Moving range computed over the full period for plotting
                - dict: mR limits {"mR_xmr_func", "mR_upper_limit"} computed from `limit_data`
                - dict: NPL limits {"y_xmr_func", "npl_upper_limit", "npl_lower_limit"} computed from `limit_data`
        """
        # Calculate moving range data for the limits (from restricted data)
        mR_data_for_limits = abs(
            limit_data[self._y_ser_name]
            - limit_data[self._y_ser_name].shift(1)
        )
        
        # Calculate moving range for the full dataset (for plotting)
        mR_data_full = abs(
            period_data[self._y_ser_name]
            - period_data[self._y_ser_name].shift(1)
        )

        # Calculate mean/median values using restricted data
        mR_xmr_func = calc_xmr_func.calc_xmr_func(mR_data_for_limits, self.xmr_function)
        y_xmr_func = calc_xmr_func.calc_xmr_func(
            limit_data[self._y_ser_name], self.xmr_function
        )

        mR_upper = mR_xmr_func * self.mR_Upper_Constant

        # Handle non-sloped case (sloped functionality can be added later if needed)
        npl_upper = y_xmr_func + (self.npl_Constant * mR_xmr_func)
        npl_lower = max(y_xmr_func - (self.npl_Constant * mR_xmr_func), 0)

        return (
            period_data,  # Return full dataset for plotting
            mR_data_full,  # Return full moving range data for plotting
            {"mR_xmr_func": mR_xmr_func, "mR_upper_limit": mR_upper},
            {
                "y_xmr_func": y_xmr_func,
                "npl_upper_limit": npl_upper,
                "npl_lower_limit": npl_lower,
            },
        )

    def _XmR_chart(self) -> tuple[Figure, dict]:
        """
        Build the XmR figure (Individuals + Moving Range) and compute SPC signals and shapes.

        Behavior:
            - Adds base traces for Individuals and mR
            - Formats axes and draws center/limit lines per period
            - Detects anomalies, long runs, and short runs
            - Populates customdata with booleans: anomaly, short run, long run
            - Rebuilds long/short run shapes directly from tooltip booleans to keep visuals aligned with tooltips
            - Adds an interactive menu to toggle Anomalies / Long Runs / Short Runs / All / None

        Returns:
            tuple:
                - Figure: The Plotly Figure containing the chart and menu
                - dict: Signals dictionary with keys:
                    - "anomalies": list of anomaly points (x, y, "High"|"Low")
                    - "long_runs": list of path point lists for each long-run highlight block
                    - "short_runs": list of path point lists for each short-run highlight block
        """
        if self._height is None:
            self._height = 600

        # Create figure with secondary y-axis
        fig_XmR = make_subplots(
            rows=2, 
            cols=1,
            subplot_titles=(self._title, "Moving Range Chart"),
            vertical_spacing=0.15,
            row_heights=[0.7,0.3]  #does nothing anyways
        )

        self.middle = []
        self.middle_mr = []
        for i, value in enumerate(self.data[self._y_ser_name]):
            if self.period_ranges:
                for start, end in self.period_ranges_index:
                    if start <= i <= end:
                        self.middle.append(self.npl_limit_values.get("y_xmr_func")[self.period_ranges_index.index((start, end))])
                        self.middle_mr.append(self.mR_limit_values.get("mR_xmr_func")[self.period_ranges_index.index((start, end))])
                        break
            else:
                self.middle.append(self.npl_limit_values.get("y_xmr_func"))
                self.middle_mr.append(self.mR_limit_values.get("mR_xmr_func"))
        
        # Create base traces
        traces = base_traces._create_base_traces(
            x_Ser=self._x_Ser,
            y_Ser=self._y_Ser,
            mR_data=self.mR_data,
            x_type=self.x_type,
            middle=self.middle,
            middle_mr=self.middle_mr,
            middle_type=self.xmr_function,
        )

        # Add traces to subplots
        fig_XmR.add_trace(traces[0], row=1, col=1)  # Individual values chart
        fig_XmR.add_trace(traces[1], row=2, col=1)  # Moving range chart

        individual_values_trace = fig_XmR.data[0]
        moving_range_trace = fig_XmR.data[1]

        axis_formats = axes_formats._format_XmR_axes(
            npl_upper=self.npl_limit_values.get("npl_upper_limit"),
            npl_lower=self.npl_limit_values.get("npl_lower_limit"),
            mR_upper=self.mR_limit_values.get("mR_upper_limit"),
            y_Ser=self._y_Ser,
            mR_data=self.mR_data,
            x_type=self.x_type,
            sloped=self.sloped,
            y_axis_dtick=self.y_axis_dtick,
        )
        fig_XmR.layout.xaxis = axis_formats.get("x_values")
        # fig_XmR.layout.xaxis2 = axis_formats.get("x_mR")
        fig_XmR.layout.yaxis = axis_formats.get("y_values")
        fig_XmR.layout.yaxis2 = axis_formats.get("y_mR")

        limit_line_shapes = limit_lines._create_limit_lines(
            data=self.data,
            y_xmr_func=self.npl_limit_values.get("y_xmr_func"),
            npl_upper=self.npl_limit_values.get("npl_upper_limit"),
            npl_lower=self.npl_limit_values.get("npl_lower_limit"),
            mR=self.mR_limit_values.get("mR_xmr_func"),
            mR_upper=self.mR_limit_values.get("mR_upper_limit"),
            sloped=self.sloped,
            period_ranges=self.period_ranges,
        )
        fig_XmR.layout.shapes = limit_line_shapes

        limit_line_annotations = annotations._create_limit_line_annotations(
            data=self.data,
            chart_title=self._title,
            y_xmr_func=self.npl_limit_values.get("y_xmr_func"),
            mR_upper=self.mR_limit_values.get("mR_upper_limit"),
            mR_xmr_func=self.mR_limit_values.get("mR_xmr_func"),
            npl_upper=self.npl_limit_values.get("npl_upper_limit"),
            npl_lower=self.npl_limit_values.get("npl_lower_limit"),
            y_name=self._y_ser_name,
            sloped=self.sloped,
            # period_ranges=self.period_ranges,
        )
        fig_XmR.layout.annotations = limit_line_annotations

        fig_XmR, anomalies = signals._anomalies(
            fig=fig_XmR,
            npl_upper=self.npl_limit_values.get("npl_upper_limit"),
            npl_lower=self.npl_limit_values.get("npl_lower_limit"),
            mR_upper=self.mR_limit_values.get("mR_upper_limit"),
            sloped=self.sloped,
            period_ranges=self.period_ranges,
        )
        long_run_shapes, long_runs, long_runs_mask = signals._long_run_test(
            fig=fig_XmR,
            x_type=self.x_type,
            y_xmr_func=self.npl_limit_values.get("y_xmr_func"),
            sloped=self.sloped,
            period_ranges=self.period_ranges,
            period_ranges_index=self.period_ranges_index,
        )

        short_run_shapes, short_runs = signals._short_run_test(
            fig_XmR,
            x_type=self.x_type,
            npl_upper=self.npl_limit_values.get("npl_upper_limit"),
            npl_lower=self.npl_limit_values.get("npl_lower_limit"),
            y_xmr_func=self.npl_limit_values.get("y_xmr_func"),
            sloped=self.sloped,
            period_ranges=self.period_ranges,
            period_ranges_index=self.period_ranges_index,
        )

        # Get run identifiers for tooltips from signals.py
        is_anomaly, is_short_run, is_long_run = _get_run_identifiers_for_tooltips(
            fig=fig_XmR,
            x_type=self.x_type,
            y_xmr_func=self.npl_limit_values.get("y_xmr_func"),
            npl_upper=self.npl_limit_values.get("npl_upper_limit"),
            npl_lower=self.npl_limit_values.get("npl_lower_limit"),
            sloped=self.sloped,
            period_ranges=self.period_ranges,
            period_ranges_index=self.period_ranges_index,
            long_runs_mask=long_runs_mask,
        )

        # Update the existing traces with boolean data for tooltips
        if len(fig_XmR.data) >= 2:
            # Update individual values chart (first trace)
            fig_XmR.data[0].customdata = np.stack((
                self.middle, 
                self.middle_mr, 
                is_anomaly, 
                is_short_run, 
                is_long_run
            ), axis=-1)
            
            # Update moving range chart (second trace)  
            fig_XmR.data[1].customdata = np.stack((
                self.middle_mr,
                self.middle, 
                is_anomaly, 
                is_short_run, 
                is_long_run
            ), axis=-1)
            
            # Update hovertemplates to include boolean fields
            if self.x_type == "date_time":
                base_template = (
                    "<b>Date:</b> %{x|%Y-%m-%d}<br>"
                    "<b>Value:</b> %{y:,.2f}<br>"
                    f"<b>{self.xmr_function.capitalize()}:</b> "
                    "%{customdata[0]:,.2f}<br>"
                    "<b>Is Anomaly:</b> %{customdata[2]}<br>"
                    "<b>Is Short Run:</b> %{customdata[3]}<br>"
                    "<b>Is Long Run:</b> %{customdata[4]}<br>"
                    "<extra></extra>"
                )
                fig_XmR.data[0].hovertemplate = base_template
                fig_XmR.data[1].hovertemplate = base_template.replace("Value", "Range")
            elif self.x_type == "numeric":
                base_template = (
                    "<b>Position:</b> %{x}<br>"
                    "<b>Value:</b> %{y:,.2f}<br>"
                    f"<b>{self.xmr_function.capitalize()}:</b> "
                    "%{customdata[0]:,.2f}<br>"
                    "<b>Is Anomaly:</b> %{customdata[2]}<br>"
                    "<b>Is Short Run:</b> %{customdata[3]}<br>"
                    "<b>Is Long Run:</b> %{customdata[4]}<br>"
                    "<extra></extra>"
                )
                fig_XmR.data[0].hovertemplate = base_template
                fig_XmR.data[1].hovertemplate = base_template.replace("Value", "Range")
            else:  # categorical
                base_template = (
                    "<b>Category:</b> %{x}<br>"
                    "<b>Value:</b> %{y:,.2f}<br>"
                    f"<b>{self.xmr_function.capitalize()}:</b> "
                    "%{customdata[0]:,.2f}<br>"
                    "<b>Is Anomaly:</b> %{customdata[2]}<br>"
                    "<b>Is Short Run:</b> %{customdata[3]}<br>"
                    "<b>Is Long Run:</b> %{customdata[4]}<br>"
                    "<extra></extra>"
                )
                fig_XmR.data[0].hovertemplate = base_template
                fig_XmR.data[1].hovertemplate = base_template.replace("Value", "Range")

        # Rebuild long-run shapes from the exact tooltip mask so shapes/paths include
        # the first index after a break when flagged (e.g., 2024-03-22)
        rebuilt_long_shapes, rebuilt_long_paths = signals._build_long_run_shapes_from_flags(
            fig=fig_XmR,
            is_long_run=is_long_run,
            y_xmr_func=self.npl_limit_values.get("y_xmr_func"),
            x_type=self.x_type,
            sloped=self.sloped,
            period_ranges_index=self.period_ranges_index,
        )

        # Rebuild short-run shapes from tooltip mask to ensure visual alignment
        rebuilt_short_shapes, rebuilt_short_paths = signals._build_short_run_shapes_from_flags(
            fig=fig_XmR,
            is_short_run=is_short_run,
            x_type=self.x_type,
        )

        fig_XmR = menus._menu(
            fig=fig_XmR,
            limit_lines=limit_line_shapes,
            limit_line_annotations=limit_line_annotations,
            long_run_shapes=rebuilt_long_shapes,
            short_run_shapes=rebuilt_short_shapes,
        )

        # Sync horizontal zoom between main chart and MR chart
        fig_XmR.update_layout(
            plot_bgcolor="white",
            font={"size": 10},
            showlegend=False,
            height=self._height,
            hovermode="x",
            xaxis2={"matches": "x"},  # Sync x-axis zoom with main chart
        )

        signals_dict = {
            "anomalies": anomalies,
            "long_runs": rebuilt_long_paths if rebuilt_long_paths else long_runs,
            "short_runs": rebuilt_short_paths if rebuilt_short_paths else short_runs,
        }

        return fig_XmR, signals_dict
