# # ============================================================
# # FILE: services/anamoly_detection.py
# # ============================================================

# # %pip install ruptures

# # Databricks-ready
# import pandas as pd
# import numpy as np
# from statsmodels.tsa.stattools import adfuller, kpss
# from statsmodels.tsa.seasonal import STL
# from sklearn.neighbors import LocalOutlierFactor
# from scipy.stats import ttest_ind
# import ruptures as rpt
# import warnings
# # warnings.filterwarnings("ignore")
# from config.config import Config


# # ============================================================
# # PREPROCESSING
# # ============================================================

# def preprocess_timeseries(df, date_col, value_col, freq="D", smooth=True, verbose=False):
#     """
#     Clean and prepare time series for anomaly detection.
#     Ensures continuous index, fills gaps and smooths missing data.
#     """
#     df = df.copy()
#     df[date_col] = pd.to_datetime(df[date_col])
#     df = df.sort_values(date_col).drop_duplicates(subset=[date_col])

#     # Infer frequency if not provided
#     if freq is None:
#         inferred = pd.infer_freq(df[date_col])
#         if inferred is None:
#             raise ValueError("Could not infer frequency. Please provide 'freq' explicitly.")
#         freq = inferred

#     # Build complete date range
#     full_range = pd.date_range(start=df[date_col].min(), end=df[date_col].max(), freq=freq)

#     # Reindex with full range
#     df = df.set_index(date_col).reindex(full_range)
#     df.index.name = date_col

#     # Fill missing values with 0
#     missing_before = df[value_col].isna().sum()
#     df[value_col] = df[value_col].fillna(0)
#     missing_after = df[value_col].isna().sum()

#     # Optional smoothing (rolling median to reduce noise)
#     if smooth:
#         df[value_col] = df[value_col].rolling(3, center=True, min_periods=1).median()

#     if verbose:
#         print(f"🧹 Preprocessing done: {missing_before} → {missing_after} NaNs, freq={freq}")

#     return (freq,df.reset_index())


# # ============================================================
# # POINT ANOMALY DETECTION METHODS
# # ============================================================

# def detect_iqr(df, value_col, multiplier=1.5):
#     """Detect outliers using Interquartile Range (IQR)"""
#     Q1 = df[value_col].quantile(0.25)
#     Q3 = df[value_col].quantile(0.75)
#     IQR = Q3 - Q1
#     lower = Q1 - multiplier * IQR
#     upper = Q3 + multiplier * IQR
#     df['iqr_flag'] = ((df[value_col] < lower) | (df[value_col] > upper)).astype(int)
#     return df

# def detect_zscore(df, value_col, threshold=3.0):
#     """Detect outliers using Z-score"""
#     window = len(df[value_col])
#     mean = df[value_col].rolling(window=window, min_periods = 1, closed='left').mean()
#     std = df[value_col].rolling(window=window, min_periods = 1, closed='left').std()
#     df['zscore'] = (df[value_col] - mean) / std
#     df['zscore_flag'] = (np.abs(df['zscore']) > threshold).astype(int)
#     return df

# def detect_lof(df, value_col, n_neighbors=20, contamination=0.05):
#     """Detect outliers using Local Outlier Factor (LOF)"""
#     lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
#     values = df[[value_col]].values
#     preds = lof.fit_predict(values)
#     df['lof_flag'] = np.where(preds == -1, 1, 0)
#     return df

# ### cp detect

# # ============================================================
# # 4️⃣ CHANGE POINT DETECTION METHODS
# # ============================================================
# def detect_bayesian_changepoint(df, value_col, pen=5):
#     """
#     Detects mean/variance shifts using Bayesian change point detection.
#     Uses ruptures library (PELT algorithm).
#     """
#     algo = rpt.Pelt(model="rbf").fit(df[value_col].values)
#     result = algo.predict(pen=pen)
#     df['bayes_flag'] = 0
#     for idx in result[:-1]:  # skip last segment endpoint
#         if idx < len(df):
#             df.loc[idx, 'bayes_flag'] = 1
#     return df


# def rolling_window_shift_detection(df, date_col, value_col, window=30, detection_size=5, alpha=0.05, threshold=0.6):
#     """
#     Detects level shifts by comparing rolling windows
#     using a two-sample t-test.
#     """
#     x = df[value_col]
#     x = np.asarray(x, dtype=float)
#     n = len(x)

#     p_values = np.ones(n)
#     shift_points = np.zeros(n)
#     left = x[n - window: n - detection_size]

#     for i in range(0, detection_size):
#         right = x[n-detection_size+i : n]

#         # Conduct two-sample t-test
#         stat, p = ttest_ind(left, right, equal_var=False)
#         p_values[n-detection_size+i] = p

#         # If p-value < alpha, significant level shift detected
#         if p < alpha:
#             shift_points[n-detection_size+i] = 1

#     df['shift_pvalues'] = p_values
#     df['shift_flag'] = 0
#     df['shift_date'] = pd.NaT

#     if shift_points[-detection_size:].sum() / detection_size > threshold:
#         df.loc[df.index[-1], "shift_flag"] = 1
#         df.loc[df.index[-1], "shift_date"] =  df.loc[df.index[-detection_size], date_col]
#         df.at[df.index[-1], "shift_pvalues"] = ",".join([f"{num:.3f}" for num in p_values[-detection_size:-1]])

#     return df

# ### test

# # 1) Magnitude Test (Z-score vs rolling mean/std)
# def magnitude_test(df, value_col, window=30, threshold=3):
#     df = df.copy()
#     rolling_mean = df[value_col].rolling(window=window, min_periods=1, closed="left").mean()
#     rolling_std = df[value_col].rolling(window=window, min_periods=1, closed="left").std().replace(0, np.nan)
#     z_score = (df[value_col] - rolling_mean) / rolling_std
#     df[f'{value_col}_result_magnitude'] = (abs(z_score) > threshold).astype(int)
#     return df

# # 2) Contextual Consistency Test (check deviation locally instead of globally)
# def contextual_consistency_test(df, value_col, window=3, threshold=50):
#     df = df.copy()
#     local_mean = df[value_col].rolling(window=window, min_periods=1, closed="left").mean()
#     local_perc = (df[value_col] - local_mean) / local_mean * 100
#     df[f'{value_col}_result_contextual'] = (abs(local_perc) > threshold).astype(int)
#     return df

# # 3) Persistence Test (check consecutive outliers)
# def persistence_test(df, value_col, persistence_window=6):
#     df = df.copy()
#     df['rolling_outlier_sum'] = df['is_outlier'].rolling(window=persistence_window, min_periods=1, closed="left").sum()
#     df[f'{value_col}_result_persistence'] = ((df['is_outlier'] == 1) & (df['rolling_outlier_sum'] < (persistence_window // 2))).astype(int)
#     df.drop(columns=['rolling_outlier_sum'], inplace=True)
#     return df

# # 4) Seasonality Test (compare same period lag, e.g. 7 days)
# def seasonality_test(df, value_col, seasonal_period=None, threshold=50):

#     if seasonal_period is None:
#         autocorr = [df[value_col].autocorr(lag) for lag in range(1,31)] # check up to a lag of 30
#         seasonal_period =  max(2, np.argmax(autocorr) + 1)

#     df['base_value'] = df[value_col].shift(seasonal_period)
#     df['seasonal_diff'] = df[value_col].diff(seasonal_period) / df['base_value'] * 100
#     df[f'{value_col}_result_seasonality'] = (abs(df['seasonal_diff']) > threshold).astype(int)
#     df.drop(columns=['base_value','seasonal_diff'], inplace=True)
#     return df

# # 5) Robustness Check (re-run with noise and confirm stability)
# def robustness_check(df, value_col, noise_std=0.01, runs=5, window=30, threshold=3):
#     df = df.copy()
#     consistent_outliers = np.zeros(len(df))

#     for _ in range(runs):
#         noisy_series = df[value_col] + np.random.normal(0, noise_std, len(df))
#         rolling_mean = noisy_series.rolling(window=window, min_periods=1, closed="left").mean()
#         rolling_std = noisy_series.rolling(window=window, min_periods=1, closed="left").std().replace(0, np.nan)
#         z_score = (noisy_series - rolling_mean) / rolling_std
#         detected = (abs(z_score) > threshold).astype(int)
#         consistent_outliers += detected.values

#     # Keep only if anomaly persists across >50% runs
#     df[f'{value_col}_result_robustness'] = (consistent_outliers > runs / 2).astype(int)
#     return df

# # 6) Maxima/Minima Test (check if point is local maxima/minima for a window of points seen so far)
# def maxima_minima_test(df, value_col, window=15):
#     df = df.copy()
#     df['is_maxima'] = df[value_col] > df[value_col].rolling(window=window, min_periods=1, closed='left').max()
#     df['is_minima'] = df[value_col] < df[value_col].rolling(window=window, min_periods=1, closed='left').min()
#     df[f'{value_col}_result_maxima_minima'] = ((df['is_maxima']) | (df['is_minima'])).astype(int)
#     df.drop(columns=['is_maxima', 'is_minima'], inplace=True)
#     return df


# def detect_trend_reversal(df, value_col, lookback=5, mode="strict", min_change_pct=1.0):
#     """
#     Detects if today's value reverses the recent trend.

#     Parameters
#     ----------
#     x : list or array
#         Time series values (must have length >= lookback + 1).
#     lookback : int
#         Number of previous points to assess trend.
#     mode : str
#         "strict"  -> every day must be higher/lower than previous
#         "smooth"  -> mean slope must be positive/negative

#     Returns
#     -------
#     dict with:
#         reversal : bool
#         previous_trend : "up", "down", or "none"
#         message : explanation
#     """
#     x = df[value_col]
#     x = np.asarray(x, dtype=float)
#     past_trend = "None"
#     reversal = False

#     past = x[-(lookback+1):-1]   # past N values
#     today = x[-1]                # today's value

#     # ---------------------------
#     # 1. Determine past trend
#     # ---------------------------
#     if mode == "strict":
#         if np.all(np.diff(past) < 0):
#             past_trend = "down"
#         elif np.all(np.diff(past) > 0):
#             past_trend = "up"

#     elif mode == "smooth":
#         slope = np.polyfit(np.arange(len(past)), past, 1)[0]
#         if slope > 0:
#             past_trend = "up"
#         elif slope < 0:
#             past_trend = "down"

#     else:
#         raise ValueError("mode must be 'strict' or 'smooth'.")

#     # ---------------------------
#     # 2. Check for reversal today
#     # ---------------------------
#     if past_trend == "down" and today > past[-1]:
#         reversal = True
#     elif past_trend == "up" and today < past[-1]:
#         reversal = True

#     df['past_trend'] = past_trend
#     df['reversal'] = reversal
#     return df

# ### perc change

# # Percentage Change Calculations
# def calculated_perc(df, value_col, calculation_lookback):
#     df = df.copy()
#     for n in calculation_lookback:
#         local_mean = df[value_col].rolling(window=n, min_periods=1, closed="left").mean()
#         df[f'{value_col}_perc_change_{n}'] = ((df[value_col] - local_mean) / local_mean * 100).round(3)
#     return df


# ### ensemble

# # ============================================================
# # 5️⃣ ENSEMBLE COMBINATION LOGIC
# # ============================================================

# def ensemble_decision(df):
#     """
#     Combine multiple methods using ensemble voting.
#     - If ≥2 point methods agree → outlier.
#     - If ≥1 change method detects → change point.
#     """
#     point_methods = ['iqr_flag', 'zscore_flag', 'lof_flag']
#     change_methods = ['shift_flag', 'bayes_flag']

#     df['is_outlier'] = (df[point_methods].sum(axis=1) >= 2).astype(int)
#     df['is_changepoint'] = (df[change_methods].sum(axis=1) >= 1).astype(int)

#     return df

# # ============================================================
# # 5️⃣ PIPELINE WRAPPER FUNCTION
# # ============================================================

# def hybrid_ensemble_pipeline(df, date_col, value_col, freq=None, window_size=30, lookback=0, threshold=3, seasonal_period=None, smooth=False, verbose=False, config_profile='relaxed'):
#     """
#     Hybrid pipeline:
#     1. Makes data stationary
#     2. Detects outliers (IQR, Z-score, LOF)
#     3. Detects change points (CUSUM, Bayesian)
#     4. Combines into final summary
#     """

#     # Preprocess
#     print("📦 Step 1: Preprocessing...")
#     df = df.sort_values(date_col).reset_index(drop=True)

#     # Select rolling window slice
#     if lookback > 0:
#         end_idx = len(df) - lookback
#     else:
#         end_idx = len(df)
#     start_idx = max(0, end_idx - window_size)
#     df_window = df.iloc[start_idx:end_idx].copy()
#     print("📦 Step 1: Preprocessing Window Data Selected...")

#     freq, df_clean = preprocess_timeseries(df_window, date_col, value_col, freq=freq, smooth=False, verbose=verbose)
#     print("📦 Step 1: Preprocessing Data is clean...")

#         ## call get_params here to get input values

#     print('inferred_frequency:', freq)
#     frequency = 'monthly' if freq in ['M', 'ME', 'MS'] else (
#         'daily' if freq == 'D'
#          else 'weekly'
#     )

#     configs = get_params(config_profile, frequency)
#     print("📦 Step 1: Data Preprocessing is Completed...")

#     # Outlier detection
#     print("🔍 Step 2: Point Anomaly Detection...")
#     point_methods = [
#         (detect_iqr, 'iqr'),
#         (detect_zscore, 'z_score'),
#         (detect_lof, 'lof')
#     ]
#     df_out = df_clean.copy()
#     for func, key in point_methods:
#         params = configs[key]
#         df_out = func(df_out, value_col, **params)
#         print(f"🔍 Step 2: Point Anomaly Detection...{key} done")

#     rolling_params = configs['rolling_window_shift']
#     df_cusum = rolling_window_shift_detection(df_out, date_col, value_col, **rolling_params)
#     print("📈 Step 3: Change Point Detection...rolling window done")

#     bayes_params = configs['change_point']
#     df_bayes = detect_bayesian_changepoint(df_cusum, value_col, **bayes_params)
#     print("📈 Step 3: Change Point Detection...bayes done")

#     # Ensemble Decisions
#     print("🤖 Step 4: Ensemble Decision...")
#     df_final = ensemble_decision(df_bayes)
#     df_final.loc[df_final.index[:-1], "is_outlier"] = 0
#     df_final.loc[df_final.index[:-5], "is_changepoint"] = 0
#     print("🤖 Step 4: Ensemble Decision done")

#     # Testing Framework
#     print("🤖 Step 5: Testing Framework...")
#     test_methods = [
#         (magnitude_test, 'magnitude_test'),
#         (contextual_consistency_test, 'contextual_consistency'),
#         (persistence_test, 'persistence_test'),
#         (seasonality_test, 'seasonal_test'),
#         (robustness_check, 'robustness_test'),
#         (maxima_minima_test, 'maxima_minima'),
#         (detect_trend_reversal, 'trend_reversal')
#     ]
#     for func, key in test_methods:
#         params = configs[key]
#         df_final = func(df_final, value_col, **params)
#     print("🤖 Step 5: Testing Framework done")

#     # Summary
#     df_final = calculated_perc(df_final, value_col, configs['perc_calculation']['calculation_lookback'])

#     print("✅ Ensemble Detection Complete.")
#     return df_final

# ## dataset loading

# datasets = ["olist_orders_dataset.csv", "olist_order_payments_dataset.csv",
# "olist_geolocation_dataset.csv","olist_customers_dataset.csv"]

# path = "/Users/harshalkothawade/.cache/kagglehub/datasets/olistbr/brazilian-ecommerce/versions/2/"

# orders = pd.read_csv(path+datasets[0])
# payments = pd.read_csv(path+datasets[1])
# # location = pd.read_csv(path+datasets[2])
# customer = pd.read_csv(path+datasets[3])
# # items = pd.read_csv(path+'olist_order_items_dataset.csv')

# payments_group = payments.groupby('order_id').sum().reset_index()
# payments_group['order_id'].value_counts()

# df_merged = pd.merge(orders,payments_group,on='order_id')
# df_merged = pd.merge(df_merged,customer,on='customer_id')

# df_merged['order_purchase_timestamp'] = pd.to_datetime(df_merged['order_purchase_timestamp']).dt.date

# # Group by 'Department' and aggregate 'Salary' with sum and 'Employee' with count
# df_opt = df_merged.groupby('order_purchase_timestamp').agg(
#     total_orders=('order_id', 'count'),
#     total_payments=('payment_value', 'sum')
# ).reset_index()
# df_opt['order_purchase_timestamp'] = pd.to_datetime(df_opt['order_purchase_timestamp'])
# df_start = df_opt[(df_opt['order_purchase_timestamp']>'2017-01-01') & (df_opt['order_purchase_timestamp']<'2018-08-01')].reset_index(drop=True)

# ## price

# final_summary_price = []

# if __name__ == "__main__":
#     np.random.seed(42)

#     date_col = 'order_purchase_timestamp'
#     freq = None
#     value_col = 'total_payments'
#     df_input = df_start[[date_col, value_col]]
#     window = 30
#     # z_score = 3
#     ds = 200

#     df_input[date_col] = pd.to_datetime(df_input[date_col])
#     df_input = df_input.sort_values(date_col).drop_duplicates(subset=[date_col])

#     for i in range(df_input.shape[0]-ds, df_input.shape[0]):

#         df_temp = df_input[:i+1]

#         df_summary = hybrid_ensemble_pipeline_2(df_temp, date_col, value_col, freq, window, verbose=False)
#         final_summary_price.append(df_summary.iloc[-1])

#         # In Databricks
#         # display(df_summary.tail(1))

# final_summary_price = pd.DataFrame(final_summary_price).reset_index(drop=True)
# final_merge_price = pd.merge(df_start, final_summary_price, 'left', [date_col,value_col])

# # final_merge_price.to_csv('anomaly_detection_payments.csv', index=False)        




# ============================================================================
# FILE: services/anomaly_detection.py (UPDATED - Modularized)
# ============================================================================
"""
Anomaly detection service with hybrid ensemble pipeline.
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings("ignore")

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False
    print("Warning: ruptures not installed. Bayesian change point detection disabled.")


class AnomalyDetectionService:
    """Service for detecting anomalies in time series data."""
    
    @staticmethod
    def preprocess_timeseries(
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        freq: str = "D",
        smooth: bool = False
    ) -> Tuple[str, pd.DataFrame]:
        """Clean and prepare time series."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).drop_duplicates(subset=[date_col])
        
        # Build complete date range
        full_range = pd.date_range(
            start=df[date_col].min(),
            end=df[date_col].max(),
            freq=freq
        )
        
        df = df.set_index(date_col).reindex(full_range)
        df.index.name = date_col
        df[value_col] = df[value_col].fillna(0)
        
        if smooth:
            df[value_col] = df[value_col].rolling(3, center=True, min_periods=1).median()
        
        return freq, df.reset_index()
    
    @staticmethod
    def detect_iqr(df: pd.DataFrame, value_col: str, multiplier: float = 1.5) -> pd.DataFrame:
        """Detect outliers using IQR."""
        Q1 = df[value_col].quantile(0.25)
        Q3 = df[value_col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        df['iqr_flag'] = ((df[value_col] < lower) | (df[value_col] > upper)).astype(int)
        return df
    
    @staticmethod
    def detect_zscore(df: pd.DataFrame, value_col: str, threshold: float = 3.0) -> pd.DataFrame:
        """Detect outliers using Z-score."""
        window = len(df[value_col])
        mean = df[value_col].rolling(window=window, min_periods=1, closed='left').mean()
        std = df[value_col].rolling(window=window, min_periods=1, closed='left').std()
        df['zscore'] = (df[value_col] - mean) / std
        df['zscore_flag'] = (np.abs(df['zscore']) > threshold).astype(int)
        return df
    
    @staticmethod
    def detect_lof(
        df: pd.DataFrame,
        value_col: str,
        n_neighbors: int = 20,
        contamination: float = 0.05
    ) -> pd.DataFrame:
        """Detect outliers using LOF."""
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        values = df[[value_col]].values
        preds = lof.fit_predict(values)
        df['lof_flag'] = np.where(preds == -1, 1, 0)
        return df
    
    @staticmethod
    def detect_bayesian_changepoint(
        df: pd.DataFrame,
        value_col: str,
        pen: int = 5
    ) -> pd.DataFrame:
        """Detect change points using Bayesian method."""
        if not HAS_RUPTURES:
            df['bayes_flag'] = 0
            return df
        
        algo = rpt.Pelt(model="rbf").fit(df[value_col].values)
        result = algo.predict(pen=pen)
        df['bayes_flag'] = 0
        for idx in result[:-1]:
            if idx < len(df):
                df.loc[idx, 'bayes_flag'] = 1
        return df
    
    @staticmethod
    def rolling_window_shift_detection(
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        window: int = 30,
        detection_size: int = 5,
        alpha: float = 0.05,
        threshold: float = 0.6
    ) -> pd.DataFrame:
        """Detect level shifts using rolling window."""
        x = df[value_col].values.astype(float)
        n = len(x)
        
        p_values = np.ones(n)
        shift_points = np.zeros(n)
        left = x[n - window: n - detection_size]
        
        for i in range(0, detection_size):
            right = x[n - detection_size + i: n]
            stat, p = ttest_ind(left, right, equal_var=False)
            p_values[n - detection_size + i] = p
            
            if p < alpha:
                shift_points[n - detection_size + i] = 1
        
        df['shift_pvalues'] = p_values
        df['shift_flag'] = 0
        df['shift_date'] = pd.NaT
        
        if shift_points[-detection_size:].sum() / detection_size > threshold:
            df.loc[df.index[-1], "shift_flag"] = 1
            df.loc[df.index[-1], "shift_date"] = df.loc[df.index[-detection_size], date_col]
        
        return df
    
    @staticmethod
    def ensemble_decision(df: pd.DataFrame) -> pd.DataFrame:
        """Combine multiple detection methods."""
        point_methods = ['iqr_flag', 'zscore_flag', 'lof_flag']
        change_methods = ['shift_flag', 'bayes_flag']
        
        df['is_outlier'] = (df[point_methods].sum(axis=1) >= 2).astype(int)
        df['is_changepoint'] = (df[change_methods].sum(axis=1) >= 1).astype(int)
        
        return df
    
    @staticmethod
    def calculate_perc_change(
        df: pd.DataFrame,
        value_col: str,
        lookback_periods: List[int] = [7, 14, 30]
    ) -> pd.DataFrame:
        """Calculate percentage changes for different lookback periods."""
        df = df.copy()
        for n in lookback_periods:
            local_mean = df[value_col].rolling(window=n, min_periods=1, closed="left").mean()
            df[f'{value_col}_perc_change_{n}'] = (
                (df[value_col] - local_mean) / local_mean * 100
            ).round(3)
        return df
    
    @staticmethod
    def hybrid_ensemble_pipeline(
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        freq: Optional[str] = None,
        window_size: int = 30,
        lookback: int = 0,
        threshold: float = 3.0,
        seasonal_period: Optional[int] = None,
        smooth: bool = False,
        verbose: bool = False,
        config_profile: str = 'relaxed'
    ) -> pd.DataFrame:
        """
        Complete hybrid anomaly detection pipeline.
        
        Args:
            df: Input DataFrame
            date_col: Name of date column
            value_col: Name of value column
            freq: Time frequency ('D', 'W', 'M')
            window_size: Rolling window size
            lookback: Lookback period
            threshold: Z-score threshold
            seasonal_period: Seasonal period for decomposition
            smooth: Whether to smooth data
            verbose: Print progress
            config_profile: 'relaxed', 'moderate', or 'strict'
            
        Returns:
            DataFrame with anomaly flags and metrics
        """
        if verbose:
            print("📦 Step 1: Preprocessing...")
        
        # Preprocess
        df = df.sort_values(date_col).reset_index(drop=True)
        
        # Select window
        if lookback > 0:
            end_idx = len(df) - lookback
        else:
            end_idx = len(df)
        start_idx = max(0, end_idx - window_size)
        df_window = df.iloc[start_idx:end_idx].copy()
        
        freq, df_clean = AnomalyDetectionService.preprocess_timeseries(
            df_window, date_col, value_col, freq or 'D', smooth
        )
        
        if verbose:
            print("🔍 Step 2: Outlier Detection...")
        
        # Outlier detection
        df_out = df_clean.copy()
        df_out = AnomalyDetectionService.detect_iqr(df_out, value_col, multiplier=1.5)
        df_out = AnomalyDetectionService.detect_zscore(df_out, value_col, threshold=threshold)
        df_out = AnomalyDetectionService.detect_lof(df_out, value_col)
        
        if verbose:
            print("📈 Step 3: Change Point Detection...")
        
        # Change point detection
        df_changes = AnomalyDetectionService.rolling_window_shift_detection(
            df_out, date_col, value_col
        )
        df_changes = AnomalyDetectionService.detect_bayesian_changepoint(
            df_changes, value_col
        )
        
        if verbose:
            print("🤖 Step 4: Ensemble Decision...")
        
        # Ensemble
        df_final = AnomalyDetectionService.ensemble_decision(df_changes)
        
        # Mark only last point as potential outlier
        df_final.loc[df_final.index[:-1], "is_outlier"] = 0
        df_final.loc[df_final.index[:-5], "is_changepoint"] = 0
        
        # Calculate percentage changes
        df_final = AnomalyDetectionService.calculate_perc_change(
            df_final, value_col, [7, 14, 30]
        )
        
        if verbose:
            print("✅ Detection Complete.")
        
        return df_final


# ============================================================================
# FILE: services/pipeline_orchestrator.py (NEW)
# ============================================================================
"""
Orchestrator for complete data processing pipeline.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List
import json

from services.data_merger_service import DataMergerService
from services.anomaly_detection import AnomalyDetectionService
from utils.session_state import SessionState


class PipelineOrchestrator:
    """Orchestrate the complete data processing and anomaly detection pipeline."""
    
    @staticmethod
    def run_complete_pipeline(
        data_folder: str,
        dkl_config: Dict,
        selected_kpis: List[str],
        verbose: bool = True
    ) -> Dict:
        """
        Run complete pipeline from data loading to anomaly detection.
        
        Args:
            data_folder: Path to data files
            dkl_config: DKL configuration dictionary
            selected_kpis: List of KPI names to analyze
            verbose: Print progress
            
        Returns:
            Dictionary with results for each KPI
        """
        results = {}
        
        try:
            # Step 1: Load data files
            if verbose:
                print("📂 Loading data files...")
            dataframes = DataMergerService.load_files(data_folder)
            
            if not dataframes:
                raise ValueError("No data files found")
            
            # Step 2: Merge dataframes if needed
            if verbose:
                print("🔗 Merging dataframes...")
            
            if len(dataframes) > 1 and 'merge_instructions' in dkl_config:
                merged_df = DataMergerService.merge_dataframes(dataframes, dkl_config)
            else:
                # Use first dataframe if no merge needed
                merged_df = list(dataframes.values())[0]
            
            # Step 3: Detect date column
            if verbose:
                print("📅 Detecting date column...")
            
            date_col = dkl_config.get('date_column')
            if not date_col:
                date_col = DataMergerService.auto_detect_date_column(merged_df)
            
            if not date_col:
                raise ValueError("Could not detect date column")
            
            # Step 4: Infer granularity
            if verbose:
                print("⏱️ Inferring time granularity...")
            
            granularity = DataMergerService.infer_granularity(merged_df, date_col)
            
            # Step 5: Discover additional KPIs
            if verbose:
                print("🔍 Discovering KPIs...")
            
            discovered_kpis = DataMergerService.discover_kpis(merged_df, date_col)
            
            # Step 6: Aggregate data
            if verbose:
                print("📊 Aggregating data...")
            
            # Filter KPIs to analyze
            kpis_to_analyze = [
                kpi for kpi in discovered_kpis
                if kpi['name'] in selected_kpis
            ]
            
            aggregated_df = DataMergerService.aggregate_data(
                merged_df,
                date_col,
                kpis_to_analyze,
                granularity
            )
            
            # Step 7: Run anomaly detection for each KPI
            if verbose:
                print("🔬 Running anomaly detection...")
            
            freq_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'ME'}
            freq = freq_map.get(granularity, 'D')
            
            for kpi_name in selected_kpis:
                if kpi_name not in aggregated_df.columns:
                    continue
                
                if verbose:
                    print(f"   Analyzing {kpi_name}...")
                
                try:
                    anomaly_results = AnomalyDetectionService.hybrid_ensemble_pipeline(
                        df=aggregated_df[[date_col, kpi_name]],
                        date_col=date_col,
                        value_col=kpi_name,
                        freq=freq,
                        window_size=30,
                        lookback=0,
                        threshold=3.0,
                        verbose=False
                    )
                    
                    results[kpi_name] = {
                        'success': True,
                        'data': anomaly_results,
                        'granularity': granularity,
                        'date_column': date_col
                    }
                    
                except Exception as e:
                    results[kpi_name] = {
                        'success': False,
                        'error': str(e)
                    }
            
            if verbose:
                print("✅ Pipeline complete!")
            
            return {
                'success': True,
                'results': results,
                'discovered_kpis': discovered_kpis,
                'granularity': granularity,
                'date_column': date_col
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

