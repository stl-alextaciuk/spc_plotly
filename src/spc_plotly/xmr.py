from pandas import DataFrame, Series, to_datetime
import pandas as pd
from numpy import abs
from spc_plotly.helpers import (
    axes_formats,
    base_traces,
    limit_lines,
    annotations,
    signals,
    menus,
)
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
    A class representing an XmR chart.

    Attributes:
        data (DataFrame): Dataframe to use for XmR chart.
        y_ser_name (str): Name of column containing values to plot on y-axis.
        x_ser_name (str): Name of column containing ordered values to plot on x-axis.
            Can be temporal (dates), numerical, or categorical.
        period_breaks (list): List of periods to break the chart on
        x_type (str): Type of x-axis data. Valid options:
            - date_time (dates and times)
            - numeric (ordered numbers)
            - categorical (ordered categories)
        date_part_resolution (str): If x_type is date_time, specify the resolution. Valid options:
            - year
            - month
            - day
            - hour
            - minute
            - custom
        custom_date_part (str): If date_part_resolution is custom, specify the format
        x_begin (str|int): Value of x_ser_name, before which the data is excluded
        x_cutoff (str|int): Value of x_ser_name, after which the data is excluded
        title (str): Custom chart title
        sloped (bool): Use sloping approach for limit values
        xmr_function (str): Use "mean" or "median" function for calculating limit values
        chart_height (int): Adjust chart height
        y_axis_dtick (float): Adjust the y-axis dticks
    """

    def __init__(
        self,
        data: DataFrame,
        y_ser_name: str,
        x_ser_name: str,
        period_breaks: list[str] = None,
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
        Initializes an XmR Chart object.

        Parameters:
            data (str): Dataframe to use for XmR chart.
            y_ser_name (int): Name of column containing values to plot on y-axis.
            x_ser_name (str): Name of column or index containing values to plot on x-axis.
                Column or index should represent a date, date/time, or a proxy for such
                (e.g., increasing integer value)
            x_begin (str): Value of x_ser_name, before which the data is excluded for purposes
                of calculating limits. If None, minimum value is set.
            x_cutoff (str): Value of x_ser_name, after which the data is excluded for purposes
                of calculating limits. If None, maximum value is set.
            date_part_resolution (str): Resolution of your data, for formatting the x-axis. Valid options:
                - year
                - month
                - day
                - hour
                - minute
                - custom
            custom_date_part (str): If you choose custom, please specify the d3 format corresponding to your data.
            title (str): Custom chart title
            sloped (bool): Use sloping approach for limit values. Only use this if your data
                is expected to increase over time (e.g., energy prices).
            xmr_function (str): Use "mean" or "median" function for calculating limit values
            chart_height (int): Adjust chart height
        """

        self.data = data
        self.xmr_function = xmr_function.lower()
        self.sloped = sloped
        self.x_type = x_type.lower()
        self.period_breaks = period_breaks
        if period_breaks:
            if x_type == "date_time":  # Convert breaks to datetime if using date_time x_type
                self.period_breaks = [pd.to_datetime(d) for d in period_breaks]
            self.period_breaks.sort()
        
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
            self._x_Ser_dt = to_datetime(self._x_Ser)
            self._x_Ser = self._x_Ser_dt.dt.strftime(self.custom_date_part)
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

        # Add selected function to mR and npl dictionaries for reference
        self.mR_limit_values["xmr_func"] = self.xmr_function
        self.npl_limit_values["xmr_func"] = self.xmr_function

        self._height = chart_height
        self.y_axis_dtick = y_axis_dtick
        self.xmr_chart, self.signals = self._XmR_chart()

    def _limits(self) -> tuple[DataFrame, Series, dict, dict]:
        """
        Calculates limits for XmR chart.

        Returns:
            tuple: A tuple containing the following;
                - pd.DataFrame: Data used to calculate limits
                - pd.Series: Data used for moving range chart
                - dict: Contains the moving range upper limit and mean/median value
                - dict: Contains the natural process limits and mean/median value
        """
        if not self.period_breaks:
            return self._calculate_period_limits(self.data)
            
#test for duplicaters too 
        test_xmr.test_sorted(self.period_breaks)
        
        # Calculate period ranges
        self.period_ranges = []
        start_date = self.x_begin or self._x_Ser.min()
        end_date = self.x_cutoff or self._x_Ser.max()
        
        # probably a better way to do this but basically creates the period lists if/else they are are dates
        if self.x_type == 'date_time':
            self.period_ranges.append((pd.to_datetime(start_date), pd.to_datetime(self.period_breaks[0])))
            for i in range(len(self.period_breaks) - 1):
                self.period_ranges.append((pd.to_datetime(self.period_breaks[i]), pd.to_datetime(self.period_breaks[i + 1])))
            self.period_ranges.append((pd.to_datetime(self.period_breaks[-1]), pd.to_datetime(end_date)))
            self._x_Ser = pd.to_datetime(self._x_Ser)
        else:
            self.period_ranges.append((start_date, self.period_breaks[0]))
            for i in range(len(self.period_breaks) - 1):
                self.period_ranges.append((self.period_breaks[i], self.period_breaks[i + 1]))
            self.period_ranges.append((self.period_breaks[-1], end_date))
        
        # Calculate limits for each period
        period_results = []
        for start, end in self.period_ranges:
            period_data = self.data.loc[
                int(self._x_Ser[self._x_Ser == start].index[0]) : int(self._x_Ser[self._x_Ser == end].index[0])
            ]
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
        Calculate limits for a single period
        """
        # Use period_data instead of self.data
        data_for_limits = period_data

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
                abs(self.data[self._y_ser_name] - self.data[self._y_ser_name].shift(1)),
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
                abs(self.data[self._y_ser_name] - self.data[self._y_ser_name].shift(1)),
                {"mR_xmr_func": mR_xmr_func, "mR_upper_limit": mR_upper},
                {
                    "y_xmr_func": y_xmr_func,
                    "npl_upper_limit": npl_upper,
                    "npl_lower_limit": npl_lower,
                },
            )

    def _XmR_chart(self) -> tuple[Figure, dict]:
        """
        Creates the XmR chart

        Returns:
            Figure: XmR chart figure object
            dict: A dictionary containing the following:
                - list: All points lying outside the limits.
                - list: List of lists, where each sublist contains points that are part
                            of a "long run", which is defined as 8 consecutive points above
                            or below the mean/median line.
                - list: List of lists, where each sublist contains points that are part
                            of a "short run", which is defined as 3 out of 4 points closer
                            to the limit lines than they are to the mean/median line.
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

        # Create base traces
        traces = base_traces._create_base_traces(
            x_Ser=self._x_Ser,
            y_Ser=self._y_Ser,
            mR_data=self.mR_data,
            x_type=self.x_type
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
            period_ranges=self.period_ranges,
            y_xmr_func=self.npl_limit_values.get("y_xmr_func"),
            npl_upper=self.npl_limit_values.get("npl_upper_limit"),
            npl_lower=self.npl_limit_values.get("npl_lower_limit"),
            mR=self.mR_limit_values.get("mR_xmr_func"),
            mR_upper=self.mR_limit_values.get("mR_upper_limit"),
            sloped=self.sloped,
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
        )
        fig_XmR.layout.annotations = limit_line_annotations

        fig_XmR, anomalies = signals._anomalies(
            fig=fig_XmR,
            npl_upper=self.npl_limit_values.get("npl_upper_limit"),
            npl_lower=self.npl_limit_values.get("npl_lower_limit"),
            mR_upper=self.mR_limit_values.get("mR_upper_limit"),
            sloped=self.sloped,
        )

        long_run_shapes, long_runs = signals._long_run_test(
            fig=fig_XmR,
            x_type=self.x_type,
            y_xmr_func=self.npl_limit_values.get("y_xmr_func"),
            sloped=self.sloped,
        )

        short_run_shapes, short_runs = signals._short_run_test(
            fig_XmR,
            x_type=self.x_type,
            npl_upper=self.npl_limit_values.get("npl_upper_limit"),
            npl_lower=self.npl_limit_values.get("npl_lower_limit"),
            y_xmr_func=self.npl_limit_values.get("y_xmr_func"),
            sloped=self.sloped,
        )

        fig_XmR = menus._menu(
            fig=fig_XmR,
            limit_lines=limit_line_shapes,
            limit_line_annotations=limit_line_annotations,
            long_run_shapes=long_run_shapes,
            short_run_shapes=short_run_shapes,
        )

        fig_XmR.update_layout(
            plot_bgcolor="white",
            font={"size": 10},
            showlegend=False,
            height=self._height,
            hovermode="x",
        )

        signals_dict = {
            "anomalies": anomalies,
            "long_runs": long_runs,
            "short_runs": short_runs,
        }

        return fig_XmR, signals_dict
