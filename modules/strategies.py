import logging
import numpy as np
import pandas as pd
import ta
import math
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

class SupertrendIndicator:
    """Supertrend indicator implementation for faster trend detection"""
    def __init__(self, period=10, multiplier=3.0):
        self.period = period
        self.multiplier = multiplier
        
    def calculate(self, df):
        """Calculate Supertrend indicator"""
        # Calculate ATR
        df['atr'] = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'], window=self.period
        )
        
        # Calculate basic upper and lower bands
        df['basic_upper'] = (df['high'] + df['low']) / 2 + (self.multiplier * df['atr'])
        df['basic_lower'] = (df['high'] + df['low']) / 2 - (self.multiplier * df['atr'])
        
        # Initialize Supertrend columns
        df['supertrend'] = np.nan
        df['supertrend_direction'] = np.nan
        df['final_upper'] = np.nan
        df['final_lower'] = np.nan
        
        # Calculate final upper and lower bands
        for i in range(self.period, len(df)):
            if i == self.period:
                # Using .loc to properly set values
                df.loc[df.index[i], 'final_upper'] = df['basic_upper'].iloc[i]
                df.loc[df.index[i], 'final_lower'] = df['basic_lower'].iloc[i]
                
                # Initial trend direction
                if df['close'].iloc[i] <= df['final_upper'].iloc[i]:
                    df.loc[df.index[i], 'supertrend'] = df['final_upper'].iloc[i]
                    df.loc[df.index[i], 'supertrend_direction'] = -1  # Downtrend
                else:
                    df.loc[df.index[i], 'supertrend'] = df['final_lower'].iloc[i]
                    df.loc[df.index[i], 'supertrend_direction'] = 1  # Uptrend
            else:
                # Calculate upper band
                if (df['basic_upper'].iloc[i] < df['final_upper'].iloc[i-1] or 
                    df['close'].iloc[i-1] > df['final_upper'].iloc[i-1]):
                    df.loc[df.index[i], 'final_upper'] = df['basic_upper'].iloc[i]
                else:
                    df.loc[df.index[i], 'final_upper'] = df['final_upper'].iloc[i-1]
                
                # Calculate lower band
                if (df['basic_lower'].iloc[i] > df['final_lower'].iloc[i-1] or 
                    df['close'].iloc[i-1] < df['final_lower'].iloc[i-1]):
                    df.loc[df.index[i], 'final_lower'] = df['basic_lower'].iloc[i]
                else:
                    df.loc[df.index[i], 'final_lower'] = df['final_lower'].iloc[i-1]
                
                # Calculate Supertrend value
                if (df['supertrend'].iloc[i-1] == df['final_upper'].iloc[i-1] and 
                    df['close'].iloc[i] <= df['final_upper'].iloc[i]):
                    df.loc[df.index[i], 'supertrend'] = df['final_upper'].iloc[i]
                    df.loc[df.index[i], 'supertrend_direction'] = -1  # Downtrend
                elif (df['supertrend'].iloc[i-1] == df['final_upper'].iloc[i-1] and 
                      df['close'].iloc[i] > df['final_upper'].iloc[i]):
                    df.loc[df.index[i], 'supertrend'] = df['final_lower'].iloc[i]
                    df.loc[df.index[i], 'supertrend_direction'] = 1  # Uptrend
                elif (df['supertrend'].iloc[i-1] == df['final_lower'].iloc[i-1] and 
                      df['close'].iloc[i] >= df['final_lower'].iloc[i]):
                    df.loc[df.index[i], 'supertrend'] = df['final_lower'].iloc[i]
                    df.loc[df.index[i], 'supertrend_direction'] = 1  # Uptrend
                elif (df['supertrend'].iloc[i-1] == df['final_lower'].iloc[i-1] and 
                      df['close'].iloc[i] < df['final_lower'].iloc[i]):
                    df.loc[df.index[i], 'supertrend'] = df['final_upper'].iloc[i]
                    df.loc[df.index[i], 'supertrend_direction'] = -1  # Downtrend
        
        return df

class TradingStrategy:
    """Base class for trading strategies"""
    def __init__(self, strategy_name):
        self.strategy_name = strategy_name
        self.risk_manager = None
        # Add cache related attributes
        self._cache = {}
        self._max_cache_entries = 10  # Limit cache size
        self._cache_expiry = 3600  # Cache expiry in seconds (1 hour)
        self._last_kline_time = None
        self._cached_dataframe = None
        
    def prepare_data(self, klines):
        """Convert raw klines to a DataFrame with OHLCV data"""
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert string values to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
            
        # Convert timestamps to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Ensure dataframe is sorted by time
        df = df.sort_values('open_time', ascending=True).reset_index(drop=True)
        
        return df
    
    def set_risk_manager(self, risk_manager):
        """Set the risk manager for the strategy"""
        self.risk_manager = risk_manager
        logger.info(f"Risk manager set for {self.strategy_name} strategy")
    
    def get_signal(self, klines):
        """
        Should be implemented by subclasses.
        Returns 'BUY', 'SELL', or None.
        """
        raise NotImplementedError("Each strategy must implement get_signal method")


class SuiDynamicGridStrategy(TradingStrategy):
    """
    Enhanced Dynamic SUI Grid Trading Strategy that adapts to market trends
    and different market conditions (bullish, bearish, and sideways).
    
    Features:
    - Dynamic position sizing based on volatility and account equity
    - Adaptive grid spacing based on market volatility
    - Asymmetric grids biased toward the trend direction
    - Automatic grid reset when price moves outside range
    - Cool-off period after consecutive losses
    - Supertrend indicator for faster trend detection
    - VWAP for sideways markets
    - Volume-weighted RSI for better signals
    - Bollinger Band squeeze detection for breakouts
    - Fibonacci level integration for support/resistance
    - Enhanced momentum filtering and multi-indicator confirmation
    - Sophisticated reversal detection
    """
    def __init__(self, 
                 grid_levels=5, 
                 grid_spacing_pct=1.2,
                 trend_ema_fast=8,
                 trend_ema_slow=21,
                 volatility_lookback=20,
                 rsi_period=14,
                 rsi_overbought=70,
                 rsi_oversold=30,
                 volume_ma_period=20,
                 adx_period=14,
                 adx_threshold=25,
                 sideways_threshold=15,
                 # SUI-specific parameters
                 volatility_multiplier=1.1,
                 trend_condition_multiplier=1.3,
                 min_grid_spacing=0.6,
                 max_grid_spacing=3.5,
                 # New parameters for enhanced features
                 supertrend_period=10,
                 supertrend_multiplier=3.0,
                 fibonacci_levels=[0.236, 0.382, 0.5, 0.618, 0.786],
                 squeeze_threshold=0.5,
                 cooloff_period=3,
                 max_consecutive_losses=2):
        
        super().__init__('SuiDynamicGridStrategy')
        # Base parameters
        self.grid_levels = grid_levels
        self.grid_spacing_pct = grid_spacing_pct
        self.trend_ema_fast = trend_ema_fast
        self.trend_ema_slow = trend_ema_slow
        self.volatility_lookback = volatility_lookback
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.volume_ma_period = volume_ma_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.sideways_threshold = sideways_threshold
        
        #SUI-specific parameters
        self.volatility_multiplier = volatility_multiplier
        self.trend_condition_multiplier = trend_condition_multiplier
        self.min_grid_spacing = min_grid_spacing
        self.max_grid_spacing = max_grid_spacing
        
        # Enhanced feature parameters
        self.supertrend_period = supertrend_period
        self.supertrend_multiplier = supertrend_multiplier
        self.fibonacci_levels = fibonacci_levels
        self.squeeze_threshold = squeeze_threshold
        self.cooloff_period = cooloff_period
        self.max_consecutive_losses = max_consecutive_losses
        
        # State variables
        self.grids = None
        self.current_trend = None
        self.current_market_condition = None
        self.last_grid_update = None
        self.consecutive_losses = 0
        self.last_loss_time = None
        self.fib_support_levels = []
        self.fib_resistance_levels = []
        self.position_size_pct = 1.0  # Default position size percentage
        
        # Cached indicators to avoid recalculation
        self._last_kline_time = None
        self._cached_dataframe = None
        self.supertrend_indicator = SupertrendIndicator(
            period=self.supertrend_period,
            multiplier=self.supertrend_multiplier
        )
        
    def prepare_data(self, klines):
        """
        Convert raw klines to a DataFrame with OHLCV data
        Overrides base method to implement enhanced caching for performance
        """
        # Generate a cache key based on first and last kline timestamps
        cache_key = None
        if len(klines) > 0:
            cache_key = f"{klines[0][0]}_{klines[-1][0]}"
        
        # Check if we can use cached data
        if cache_key:
            current_time = int(datetime.now().timestamp())
            
            # Clean up expired cache entries periodically
            if random.random() < 0.05:  # 5% chance to clean on each call
                expired_keys = []
                for k, v in self._cache.items():
                    if current_time - v.get('time', 0) > self._cache_expiry:
                        expired_keys.append(k)
                for k in expired_keys:
                    del self._cache[k]
                    logger.debug(f"Removed expired cache entry: {k}")
            
            # Look for cache entry
            if cache_key in self._cache:
                cache_entry = self._cache[cache_key]
                cache_time = cache_entry.get('time', 0)
                
                # Check if cache is still valid (not expired)
                if current_time - cache_time < self._cache_expiry:
                    logger.debug(f"Using cached data for {cache_key}")
                    return cache_entry['data']
        
        # Fall back to simple cache check if complex caching fails
        if len(klines) > 0 and self._last_kline_time == klines[-1][0]:
            if self._cached_dataframe is not None:
                return self._cached_dataframe
            
        # Otherwise prepare data normally
        df = super().prepare_data(klines)
        
        # Cache the result
        if len(klines) > 0:
            # Simple caching (backward compatible)
            self._last_kline_time = klines[-1][0]
            self._cached_dataframe = df
            
            # Enhanced caching with expiry and size management
            if cache_key:
                # Manage cache size - remove oldest entry if needed
                if len(self._cache) >= self._max_cache_entries:
                    oldest_key = min(self._cache.keys(), 
                                    key=lambda k: self._cache[k].get('time', 0))
                    del self._cache[oldest_key]
                    logger.debug(f"Cache full, removed oldest entry {oldest_key}")
                
                # Store in cache with timestamp
                self._cache[cache_key] = {
                    'data': df,
                    'time': current_time
                }
                logger.debug(f"Cached data for {cache_key}")
            
        return df
    
    def add_indicators(self, df):
        """Add technical indicators to the DataFrame with enhanced features"""
        # Trend indicators
        df['ema_fast'] = ta.trend.ema_indicator(df['close'], 
                                               window=self.trend_ema_fast)
        df['ema_slow'] = ta.trend.ema_indicator(df['close'], 
                                               window=self.trend_ema_slow)
        
        # Add Supertrend indicator for faster trend detection
        df = self.supertrend_indicator.calculate(df)
        df['trend'] = np.where(df['supertrend_direction'] == 1, 'UPTREND', 'DOWNTREND')
        
        # Momentum indicators
        df['rsi'] = ta.momentum.rsi(df['close'], window=self.rsi_period)
        
        # Volume indicators first to avoid duplicate calculation
        df['volume_ma'] = ta.trend.sma_indicator(df['volume'], 
                                                window=self.volume_ma_period)
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Volume-weighted RSI (using the volume_ratio calculated above)
        df['volume_weighted_rsi'] = df['rsi'] * df['volume_ratio']
        
        # Volatility indicators
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], 
                                                    df['close'], 
                                                    window=self.volatility_lookback)
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        # ADX for trend strength
        adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], 
                                             window=self.adx_period)
        df['adx'] = adx_indicator.adx()
        df['di_plus'] = adx_indicator.adx_pos()
        df['di_minus'] = adx_indicator.adx_neg()
        
        # Bollinger Bands
        indicator_bb = ta.volatility.BollingerBands(df['close'], 
                                                   window=20, 
                                                   window_dev=2)
        df['bb_upper'] = indicator_bb.bollinger_hband()
        df['bb_middle'] = indicator_bb.bollinger_mavg()
        df['bb_lower'] = indicator_bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Bollinger Band Squeeze detection
        df['bb_squeeze'] = df['bb_width'] < self.squeeze_threshold
        
        # MACD for additional trend confirmation
        macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        df['macd_crossover'] = np.where(
            (df['macd'].shift(1) < df['macd_signal'].shift(1)) & 
            (df['macd'] > df['macd_signal']), 
            1, np.where(
                (df['macd'].shift(1) > df['macd_signal'].shift(1)) & 
                (df['macd'] < df['macd_signal']), 
                -1, 0
            )
        )
        
        # VWAP (Volume Weighted Average Price) - calculated per day
        df['vwap'] = self.calculate_vwap(df)
        
        # Calculate Fibonacci levels based on recent swing highs and lows
        self.calculate_fibonacci_levels(df)
        
        # Market condition classification with improved detection
        df['market_condition'] = self.classify_market_condition(df)
        
        # Reversal detection indicators
        df['potential_reversal'] = self.detect_reversal_patterns(df)
        
        return df
    
    def calculate_vwap(self, df):
        """Calculate VWAP (Volume Weighted Average Price)"""
        # Get date component of timestamp for grouping
        df['date'] = df['open_time'].dt.date
        
        # Calculate VWAP for each day
        vwap = pd.Series(index=df.index)
        for date, group in df.groupby('date'):
            # Calculate cumulative sum of price * volume
            cum_vol_price = (group['close'] * group['volume']).cumsum()
            # Calculate cumulative sum of volume
            cum_vol = group['volume'].cumsum()
            # Calculate VWAP
            daily_vwap = cum_vol_price / cum_vol
            # Add to result series
            vwap.loc[group.index] = daily_vwap
            
        return vwap
    
    def calculate_fibonacci_levels(self, df):
        """Calculate Fibonacci retracement/extension levels for support and resistance"""
        if len(df) < 20:  # Need sufficient data
            return
        
        # Find recent swing high and low points
        window = min(100, len(df) - 1)  # Look back window
        price_data = df['close'].iloc[-window:]
        
        # Identify swing high and low
        swing_high = price_data.max()
        swing_low = price_data.min()
        
        # Reset fibonacci levels
        self.fib_support_levels = []
        self.fib_resistance_levels = []
        
        # Calculate levels based on trend
        latest = df.iloc[-1]
        current_price = latest['close']
        current_trend = latest['trend']
        
        if current_trend == 'UPTREND':
            # In uptrend, calculate fib retracements from low to high for support
            for fib in self.fibonacci_levels:
                level = swing_low + (swing_high - swing_low) * (1 - fib)
                if level < current_price:
                    self.fib_support_levels.append(level)
                else:
                    self.fib_resistance_levels.append(level)
                    
            # Add extension levels for resistance
            for ext in [1.272, 1.618, 2.0]:
                level = swing_low + (swing_high - swing_low) * ext
                self.fib_resistance_levels.append(level)
                
        else:  # DOWNTREND
            # In downtrend, calculate fib retracements from high to low for resistance
            for fib in self.fibonacci_levels:
                level = swing_high - (swing_high - swing_low) * fib
                if level > current_price:
                    self.fib_resistance_levels.append(level)
                else:
                    self.fib_support_levels.append(level)
                    
            # Add extension levels for support
            for ext in [1.272, 1.618, 2.0]:
                level = swing_high - (swing_high - swing_low) * ext
                self.fib_support_levels.append(level)
        
        # Sort the levels
        self.fib_support_levels.sort(reverse=True)  # Descending
        self.fib_resistance_levels.sort()  # Ascending
    
    def detect_reversal_patterns(self, df):
        """
        Enhanced reversal pattern detection
        Returns 1 for potential bullish reversal, -1 for bearish reversal, 0 for no reversal
        """
        if len(df) < 5:
            return pd.Series(0, index=df.index)
            
        # Initialize result series
        reversal = pd.Series(0, index=df.index)
        
        for i in range(4, len(df)):
            # Get relevant rows for pattern detection
            curr = df.iloc[i]
            prev1 = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            prev3 = df.iloc[i-3]
            
            # Check for bullish reversal patterns
            bullish_reversal = False
            
            # Bullish engulfing
            if (curr['close'] > curr['open'] and  # Current candle is bullish
                prev1['close'] < prev1['open'] and  # Previous candle is bearish
                curr['close'] > prev1['open'] and  # Current close above prev open
                curr['open'] < prev1['close']):  # Current open below prev close
                bullish_reversal = True
                
            # Hammer pattern (bullish)
            elif (curr['low'] < curr['open'] and
                  curr['low'] < curr['close'] and
                  (curr['high'] - max(curr['open'], curr['close'])) < 
                  (min(curr['open'], curr['close']) - curr['low']) * 2 and
                  (min(curr['open'], curr['close']) - curr['low']) > 
                  (curr['high'] - max(curr['open'], curr['close'])) * 3):
                bullish_reversal = True
                
            # RSI divergence (bullish)
            elif (prev2['low'] > prev1['low'] and  # Price making lower low
                  prev2['rsi'] < prev1['rsi'] and  # RSI making higher low
                  curr['supertrend_direction'] == 1):  # Confirmed by Supertrend
                bullish_reversal = True
                
            # Check for bearish reversal patterns
            bearish_reversal = False
            
            # Bearish engulfing
            if (curr['close'] < curr['open'] and  # Current candle is bearish
                prev1['close'] > prev1['open'] and  # Previous candle is bullish
                curr['close'] < prev1['open'] and  # Current close below prev open
                curr['open'] > prev1['close']):  # Current open above prev close
                bearish_reversal = True
                
            # Shooting star (bearish)
            elif (curr['high'] > curr['open'] and
                  curr['high'] > curr['close'] and
                  (curr['high'] - max(curr['open'], curr['close'])) > 
                  (min(curr['open'], curr['close']) - curr['low']) * 2 and
                  (curr['high'] - max(curr['open'], curr['close'])) > 
                  (min(curr['open'], curr['close']) - curr['low']) * 3):
                bearish_reversal = True
                
            # RSI divergence (bearish)
            elif (prev2['high'] < prev1['high'] and  # Price making higher high
                  prev2['rsi'] > prev1['rsi'] and  # RSI making lower high
                  curr['supertrend_direction'] == -1):  # Confirmed by Supertrend
                bearish_reversal = True
            
            # Set reversal value
            if bullish_reversal:
                reversal.iloc[i] = 1
            elif bearish_reversal:
                reversal.iloc[i] = -1
                
        return reversal
    
    def classify_market_condition(self, df):
        """
        Enhanced market condition classification with better state transitions and stability
        """
        conditions = []
        lookback_period = 10  # Use more historical data for smoother transitions
        current_condition = None
        condition_streak = 0  # Track how long we've been in a condition
        
        for i in range(len(df)):
            if i < self.adx_period:
                conditions.append('SIDEWAYS')  # Default for initial rows
                continue
                
            # Get relevant indicators
            adx = df['adx'].iloc[i]
            di_plus = df['di_plus'].iloc[i]
            di_minus = df['di_minus'].iloc[i]
            rsi = df['rsi'].iloc[i]
            bb_width = df['bb_width'].iloc[i]
            supertrend_dir = df['supertrend_direction'].iloc[i] if i >= self.supertrend_period else 0
            macd_crossover = df['macd_crossover'].iloc[i] if 'macd_crossover' in df else 0
            
            # Get average ADX for more stability
            lookback = min(lookback_period, i)
            avg_adx = df['adx'].iloc[i-lookback:i+1].mean() if i >= lookback else adx
            
            # Calculate the strength of each potential market condition
            bullish_strength = 0
            bearish_strength = 0
            sideways_strength = 0
            
            # ADX trending strength (0-100)
            trend_strength = min(100, adx * 2)  # Normalize to 0-100 scale
            
            # Directional bias (-100 to +100, negative = bearish, positive = bullish)
            if di_plus + di_minus > 0:  # Avoid division by zero
                directional_bias = 100 * (di_plus - di_minus) / (di_plus + di_minus)
            else:
                directional_bias = 0
                
            # Supertrend confirmation
            supertrend_bias = 100 if supertrend_dir > 0 else -100 if supertrend_dir < 0 else 0
            
            # RSI bias (normalized to -100 to +100)
            rsi_bias = (rsi - 50) * 2  # 0 = neutral, +100 = extremely bullish, -100 = extremely bearish
            
            # Combine for final bias score
            bias_score = (directional_bias + supertrend_bias + rsi_bias) / 3
            
            # Get the current row for indicator checks
            current_row = df.iloc[i]
            
            # Calculate condition strengths with enhanced filter requirements
            # This makes trend classification more strict, leading to higher quality signals
            if bias_score > 40 and trend_strength > 60:  # Strong bullish - increased thresholds
                bullish_strength = trend_strength
                
                # Confirm with price action relative to EMAs
                if current_row['close'] > current_row['ema_fast'] > current_row['ema_slow']:
                    bullish_strength += 15  # Additional confirmation from EMA alignment
                
                # Determine if extreme bullish - stricter requirements
                if bias_score > 70 and trend_strength > 75 and adx > self.adx_threshold * 1.8:
                    bullish_strength += 25  # Extra boost for extreme bullish
            
            if bias_score < -40 and trend_strength > 60:  # Strong bearish - increased thresholds
                bearish_strength = trend_strength
                
                # Confirm with price action relative to EMAs
                if current_row['close'] < current_row['ema_fast'] < current_row['ema_slow']:
                    bearish_strength += 15  # Additional confirmation from EMA alignment
                
                # Determine if extreme bearish - stricter requirements
                if bias_score < -70 and trend_strength > 75 and adx > self.adx_threshold * 1.8:
                    bearish_strength += 25  # Extra boost for extreme bearish
            
            if trend_strength < 45:  # Low trend strength = sideways - adjusted threshold
                sideways_strength = 100 - trend_strength
                
                # Add mean-reversion confirmation for stronger sideways classification
                if abs(current_row['close'] - current_row['vwap']) < current_row['atr'] * 0.5:
                    sideways_strength += 15  # Price near VWAP confirms sideways
                
            # Removed squeeze condition check
            
            # Prevent rapid condition changes by requiring larger threshold to change states
            new_condition = None
            
            # Determine the new condition based on highest strength
            if max(bullish_strength, bearish_strength, sideways_strength) == bullish_strength:
                if bullish_strength > 80:
                    new_condition = 'EXTREME_BULLISH'
                else:
                    new_condition = 'BULLISH'
            elif max(bullish_strength, bearish_strength, sideways_strength) == bearish_strength:
                if bearish_strength > 80:
                    new_condition = 'EXTREME_BEARISH'
                else:
                    new_condition = 'BEARISH'
            else:
                new_condition = 'SIDEWAYS'
            
            # Apply hysteresis to avoid rapid condition changes
            # Only change condition if new one is persistent or very strong
            if i > 0 and current_condition:
                previous_condition = conditions[i-1]
                
                # Keep current condition unless we have a strong signal to change
                # This prevents whipsaws between market states
                if previous_condition != new_condition:
                    # For a change to occur, the new condition needs to be significantly stronger
                    change_threshold = 20 if condition_streak < 3 else 10
                    
                    max_current_strength = 0
                    if previous_condition == 'BULLISH' or previous_condition == 'EXTREME_BULLISH':
                        max_current_strength = bullish_strength
                    elif previous_condition == 'BEARISH' or previous_condition == 'EXTREME_BEARISH':
                        max_current_strength = bearish_strength
                    else:  # SIDEWAYS
                        max_current_strength = sideways_strength
                    
                    max_new_strength = 0
                    if new_condition == 'BULLISH' or new_condition == 'EXTREME_BULLISH':
                        max_new_strength = bullish_strength
                    elif new_condition == 'BEARISH' or new_condition == 'EXTREME_BEARISH':
                        max_new_strength = bearish_strength
                    else:  # SIDEWAYS
                        max_new_strength = sideways_strength
                    
                    # Only change if the new condition is significantly stronger
                    if max_new_strength > max_current_strength + change_threshold:
                        conditions.append(new_condition)
                        current_condition = new_condition
                        condition_streak = 1
                    else:
                        conditions.append(previous_condition)
                        current_condition = previous_condition
                        condition_streak += 1
                else:
                    conditions.append(previous_condition)
                    current_condition = previous_condition
                    condition_streak += 1
            else:
                conditions.append(new_condition)
                current_condition = new_condition
                condition_streak = 1
        
        return pd.Series(conditions, index=df.index)
    
    def calculate_dynamic_position_size(self, df, base_position=1.0):
        """
        Calculate dynamic position size based on volatility and market condition
        
        Args:
            df: DataFrame with indicators
            base_position: Base position size (1.0 = 100% of allowed position)
            
        Returns:
            float: Position size multiplier (e.g., 0.8 means 80% of base position)
        """
        latest = df.iloc[-1]
        
        # Get ATR as volatility measure
        atr_pct = latest['atr_pct']
        avg_atr_pct = df['atr_pct'].tail(20).mean()
        
        # Base position sizing on volatility relative to average
        if atr_pct > avg_atr_pct * 1.5:
            # High volatility - reduce position size
            volatility_factor = 0.7
        elif atr_pct < avg_atr_pct * 0.7:
            # Low volatility - increase position size slightly
            volatility_factor = 1.1
        else:
            # Normal volatility
            volatility_factor = 1.0
            
        # Adjust based on market condition
        market_condition = latest['market_condition']
        if market_condition in ['EXTREME_BULLISH', 'EXTREME_BEARISH']:
            # Extreme trend - reduce position size for safety
            condition_factor = 0.75
        elif market_condition in ['BULLISH', 'BEARISH']:
            # Clear trend - standard position
            condition_factor = 1.0
        else:  # SIDEWAYS
            # Sideways market - slightly smaller position
            condition_factor = 0.9
            
        # Calculate final position size
        position_size = base_position * volatility_factor * condition_factor
        
        # Cap the position size
        return min(max(position_size, 0.5), 1.2)  # Between 50% and 120%
    
    def calculate_dynamic_grid_levels(self, df):
        """
        Adjust grid levels count based on volatility
        """
        latest = df.iloc[-1]
        
        # Base decision on ATR percentage and BB width
        atr_pct = latest['atr_pct']
        bb_width = latest['bb_width']
        
        # Fewer levels in high volatility
        if atr_pct > 3.0 or bb_width > 0.08:
            return max(3, self.grid_levels - 2)
        # More levels in low volatility
        elif atr_pct < 1.0 or bb_width < 0.03:
            return min(7, self.grid_levels + 2)
        # Default to configured level
        else:
            return self.grid_levels
            
    def calculate_grid_spacing(self, df):
        """
        Enhanced dynamic grid spacing calculation with superior volatility handling
        and strategic spacing to improve win rate
        """
        try:
            # Get the latest row and some historical data
            latest = df.iloc[-1]
            
            # Calculate a more stable volatility measure using different timeframes
            # This helps avoid excessive reactivity to short-term volatility spikes
            atr_short = df['atr_pct'].tail(7).mean()  # Short-term ATR (7 periods)
            atr_medium = df['atr_pct'].tail(14).mean()  # Medium-term ATR (14 periods)
            atr_long = df['atr_pct'].tail(30).mean() if len(df) >= 30 else df['atr_pct'].mean()  # Longer-term ATR
            
            # Use weighted average of different ATR periods for more stability
            weighted_atr = (atr_short * 0.5) + (atr_medium * 0.3) + (atr_long * 0.2)
            
            # Base grid spacing on weighted ATR percentage with more conservative multiplier
            base_spacing = weighted_atr * self.volatility_multiplier * 1.25
            
            # Adjust based on Bollinger Band width but with more conservative scaling
            bb_width = latest['bb_width']
            bb_historical = df['bb_width'].tail(20).mean()  # Average BB width for more stability
            bb_ratio = bb_width / bb_historical if bb_historical > 0 else 1.0
            
            # More conservative BB multiplier
            bb_multiplier = min(max(bb_ratio * 4.5, 0.8), 2.8)  # Tightened range from (0.5, 3.5) to (0.8, 2.8)
            
            # Adjust based on market condition and strength of trend
            market_condition = latest['market_condition']
            adx_strength = latest['adx']
            adx_strength_factor = min(adx_strength / 25.0, 1.5)  # Cap the impact of ADX
            
            if market_condition == 'SIDEWAYS':
                # Tighter grid spacing in sideways markets for higher probability trades
                condition_multiplier = 0.7
            elif market_condition in ['BULLISH', 'BEARISH']:
                # Wider grid spacing in trending markets, adjusted by ADX strength
                condition_multiplier = self.trend_condition_multiplier * adx_strength_factor
            elif market_condition in ['EXTREME_BULLISH', 'EXTREME_BEARISH']:
                # Even wider spacing in extreme trends, adjusted by ADX strength
                condition_multiplier = self.trend_condition_multiplier * adx_strength_factor * 1.3
            else:
                condition_multiplier = 1.0
            
            # Calculate final grid spacing with weighted components
            dynamic_spacing = base_spacing * bb_multiplier * condition_multiplier
            
            # Ensure minimum and maximum spacing with tighter bounds
            # This prevents excessively wide grids that might miss trading opportunities
            min_spacing = self.min_grid_spacing * 1.2  # Slightly wider minimum
            max_spacing = self.max_grid_spacing * 0.9  # Slightly tighter maximum
            
            return min(max(dynamic_spacing, min_spacing), max_spacing)
            
        except Exception as e:
            logger.error(f"Error calculating grid spacing: {e}")
            # Return default spacing in case of error
            return self.grid_spacing_pct
    
    def calculate_grid_bias(self, df):
        """
        Calculate asymmetric grid bias based on market conditions with enhanced trend filtering
        Returns percentage of grid levels that should be above current price
        """
        latest = df.iloc[-1]
        market_condition = latest['market_condition']
        
        # Get additional confirmations for trend strength
        supertrend_dir = latest['supertrend_direction']
        adx_strength = latest['adx']
        rsi = latest['rsi']
        
        # Check if price is above/below key moving averages
        above_ema_slow = latest['close'] > latest['ema_slow']
        above_bb_middle = latest['close'] > latest['bb_middle']
        
        # Extreme bias in extreme market conditions with confirmation
        if market_condition == 'EXTREME_BULLISH' and supertrend_dir == 1 and above_ema_slow:
            return 0.85  # 85% of levels above (very strong buy bias)
        elif market_condition == 'EXTREME_BEARISH' and supertrend_dir == -1 and not above_ema_slow:
            return 0.15  # 15% of levels above (very strong sell bias)
        # Strong bias in trending markets with confirmation
        elif market_condition == 'BULLISH' and above_ema_slow and adx_strength > 25:
            return 0.75  # 75% of levels above (strong buy bias)
        elif market_condition == 'BEARISH' and not above_ema_slow and adx_strength > 25:
            return 0.25  # 25% of levels above (strong sell bias)
        # More neutral bias when trend confirmation is mixed
        elif market_condition == 'BULLISH':
            return 0.65  # 65% of levels above (moderate buy bias)
        elif market_condition == 'BEARISH':
            return 0.35  # 35% of levels above (moderate sell bias)
        # Neutral bias in sideways markets
        else:
            # Slightly adjust based on RSI for mean-reversion in sideways markets
            if rsi > 60 and above_bb_middle:
                return 0.4  # 40% of levels above (slight sell bias for overbought)
            elif rsi < 40 and not above_bb_middle:
                return 0.6  # 60% of levels above (slight buy bias for oversold)
            else:
                return 0.5  # 50% of levels above/below (neutral)
    
    def generate_grids(self, df):
        """
        Generate enhanced dynamic grid levels with asymmetric distribution
        and Fibonacci integration - WITH REVERSED GRID TYPES
        """
        # Get latest price and indicators
        latest = df.iloc[-1]
        current_price = latest['close']
        current_trend = latest['trend']
        market_condition = latest['market_condition']
        
        # Update risk manager with market condition if available
        if self.risk_manager and market_condition:
            self.risk_manager.set_market_condition(market_condition)
            self.risk_manager.update_position_sizing(self.calculate_dynamic_position_size(df))
            logger.info(f"Updated risk manager with market condition: {market_condition}")
        
        # Determine asymmetric grid bias based on market condition
        grid_bias = self.calculate_grid_bias(df)
        
        # Calculate dynamic grid spacing based on volatility
        dynamic_spacing = self.calculate_grid_spacing(df)
        
        # Adjust grid levels based on volatility
        dynamic_grid_levels = self.calculate_dynamic_grid_levels(df)
        
        # Generate grid levels
        grid_levels = []
        
        # Calculate number of levels above and below current price
        levels_above = int(dynamic_grid_levels * grid_bias)
        levels_below = dynamic_grid_levels - levels_above
        
        # Generate grid levels below current price
        for i in range(1, levels_below + 1):
            # Base grid price
            base_grid_price = current_price * (1 - (dynamic_spacing / 100) * i)
            
            # Check if any Fibonacci support level is nearby
            grid_price = base_grid_price
            for support_level in self.fib_support_levels:
                # If within 1% of a Fibonacci level, snap to it
                if abs(support_level - base_grid_price) / base_grid_price < 0.01:
                    grid_price = support_level
                    break
            
            # REVERSED: Change from BUY to SELL
            grid_levels.append({
                'price': grid_price,
                'type': 'SELL',  # REVERSED from 'BUY' to 'SELL'
                'status': 'ACTIVE',
                'created_at': latest['open_time']
            })
        
        # Generate grid levels above current price
        for i in range(1, levels_above + 1):
            # Base grid price
            base_grid_price = current_price * (1 + (dynamic_spacing / 100) * i)
            
            # Check if any Fibonacci resistance level is nearby
            grid_price = base_grid_price
            for resistance_level in self.fib_resistance_levels:
                # If within 1% of a Fibonacci level, snap to it
                if abs(resistance_level - base_grid_price) / base_grid_price < 0.01:
                    grid_price = resistance_level
                    break
            
            # REVERSED: Change from SELL to BUY
            grid_levels.append({
                'price': grid_price,
                'type': 'BUY',  # REVERSED from 'SELL' to 'BUY'
                'status': 'ACTIVE',
                'created_at': latest['open_time']
            })
        
        # Sort grid levels by price
        grid_levels.sort(key=lambda x: x['price'])
        
        return grid_levels
    
    def should_update_grids(self, df):
        """Enhanced grid reset logic"""
        if self.grids is None or len(self.grids) == 0:
            return True
            
        latest = df.iloc[-1]
        current_trend = latest['trend']
        current_market_condition = latest['market_condition']
        
        # Update risk manager with new market condition if it changed
        if self.risk_manager and self.current_market_condition != current_market_condition:
            self.risk_manager.set_market_condition(current_market_condition)
            logger.info(f"Updated risk manager with market condition: {current_market_condition}")
        
        # Update grids if trend or market condition changed significantly
        trend_change = (self.current_trend != current_trend)
        condition_change = (
            (self.current_market_condition in ['BULLISH', 'BEARISH'] and 
             current_market_condition in ['EXTREME_BULLISH', 'EXTREME_BEARISH']) or
            (self.current_market_condition in ['EXTREME_BULLISH', 'EXTREME_BEARISH'] and 
             current_market_condition in ['BULLISH', 'BEARISH'])
        )
        
        if trend_change or condition_change:
            logger.info(f"Market conditions changed. Trend: {self.current_trend}->{current_trend}, "
                       f"Condition: {self.current_market_condition}->{current_market_condition}. "
                       f"Updating grids.")
            return True
            
        # Check if price moved significantly outside grid range (auto-reset)
        current_price = latest['close']
        min_grid = min(grid['price'] for grid in self.grids)
        max_grid = max(grid['price'] for grid in self.grids)
        
        # If price is outside grid range by more than 2%, update grids
        if current_price < min_grid * 0.98 or current_price > max_grid * 1.02:
            logger.info(f"Price moved outside grid range. Updating grids.")
            return True
            
        # Check if many grid levels have been triggered
        active_grids = [grid for grid in self.grids if grid['status'] == 'ACTIVE']
        if len(active_grids) < len(self.grids) * 0.3:  # Less than 30% active
            logger.info(f"Too many grid levels have been triggered. Refreshing grid.")
            return True
            
        return False
    
    def in_cooloff_period(self, current_time):
        """Check if we're in a cool-off period after any loss"""
        if self.last_loss_time:  # If we have a recorded loss time
            try:
                # Convert from pandas timestamp if needed
                if hasattr(self.last_loss_time, 'to_pydatetime'):
                    last_loss_time = self.last_loss_time.to_pydatetime()
                else:
                    last_loss_time = self.last_loss_time
                    
                # Convert current_time from pandas timestamp if needed
                if hasattr(current_time, 'to_pydatetime'):
                    current_time = current_time.to_pydatetime()
                
                # Check if we're still in the cool-off period
                cooloff_end_time = last_loss_time + timedelta(minutes=self.cooloff_period)
                
                # If we're still in cooloff period, return True
                if current_time < cooloff_end_time:
                    # Log whenever we're in a cool-off period
                    logger.debug(f"In cool-off period. Remaining time: {(cooloff_end_time - current_time).total_seconds() / 60:.1f} minutes")
                    return True
                
                # If we just exited cooloff period, refresh the system
                if not hasattr(self, '_last_cooloff_check') or self._last_cooloff_check < cooloff_end_time:
                    self._last_cooloff_check = current_time
                    self._refresh_system_after_cooloff()
                    logger.info("Cooloff period ended - system refreshed completely with all old data removed")
                
                return False
            except Exception as e:
                logger.error(f"Error in cooloff period calculation: {e}")
                # Default to False if there's an error in comparison
                return False
            
        return False
    
    def _refresh_system_after_cooloff(self):
        """
        Reset all system state after cooloff period ends.
        This clears cache, removes old data, and refreshes system to start fresh.
        """
        # Clear all caches and cached data
        self._last_kline_time = None
        self._cached_dataframe = None
        if hasattr(self, '_cache'):
            self._cache = {}
        
        # Reset all state variables
        self.grids = None
        self.current_trend = None
        self.current_market_condition = None
        self.last_grid_update = None
        self.consecutive_losses = 0  # Reset consecutive losses counter
        self.fib_support_levels = []
        self.fib_resistance_levels = []
        
        # Reset position size to default
        self.position_size_pct = 1.0
        
        # Reset any additional tracking variables
        if hasattr(self, '_last_cooloff_check'):
            delattr(self, '_last_cooloff_check')
        
        # Reset supertrend indicator to fresh state
        self.supertrend_indicator = SupertrendIndicator(
            period=self.supertrend_period,
            multiplier=self.supertrend_multiplier
        )
        
        # Send Telegram notification about the reset
        try:
            from modules.telegram_notifier import TelegramNotifier
            notifier = TelegramNotifier()
            notifier.send_message(f"🔄 *Cool-off Period Ended*\n\nBot has refreshed all systems and will resume trading with fresh data.")
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
        
        logger.info("System state completely refreshed after cooloff period: all caches cleared, state reset, bot ready for fresh start")
    
    def get_v_reversal_signal(self, df):
        """
        Enhanced V-shaped reversal detection in extreme market conditions
        WITH REVERSED SIGNALS and additional confirmation requirements for higher win rate
        """
        if len(df) < 7:  # Increased minimum data requirement
            return None
            
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        market_condition = latest['market_condition']
        potential_reversal = latest['potential_reversal']
        
        # Only look for reversals in extreme conditions
        if market_condition not in ['EXTREME_BEARISH', 'EXTREME_BULLISH']:
            return None
        
        # Additional confirmation filters to ensure high-quality reversal signals
        
        # Volume confirmation - require above average volume on reversal
        volume_confirmation = latest['volume_ratio'] > 1.5
        
        # RSI confirmation - require extreme readings
        rsi_confirmation_bullish = latest['rsi'] < 25 and prev['rsi'] < latest['rsi']  # RSI turning up from oversold
        rsi_confirmation_bearish = latest['rsi'] > 75 and prev['rsi'] > latest['rsi']  # RSI turning down from overbought
        
        # Price action confirmation - require candle close beyond key level
        if market_condition == 'EXTREME_BEARISH':
            price_confirmation = latest['close'] > latest['ema_fast'] and latest['close'] > prev['high']
        else:  # EXTREME_BULLISH
            price_confirmation = latest['close'] < latest['ema_fast'] and latest['close'] < prev['low']
            
        # Return reversal signal if detected with stronger confirmation - REVERSED signals
        if (potential_reversal == 1 and 
            market_condition == 'EXTREME_BEARISH' and 
            volume_confirmation and 
            rsi_confirmation_bullish and 
            price_confirmation):
            logger.info("Strong V-shaped bullish reversal detected in extreme bearish market with multiple confirmations - REVERSED to SELL")
            # REVERSED: return SELL instead of BUY
            return 'SELL'
        elif (potential_reversal == -1 and 
              market_condition == 'EXTREME_BULLISH' and 
              volume_confirmation and 
              rsi_confirmation_bearish and 
              price_confirmation):
            logger.info("Strong V-shaped bearish reversal detected in extreme bullish market with multiple confirmations - REVERSED to BUY")
            # REVERSED: return BUY instead of SELL
            return 'BUY'
            
        return None
    
    # SQUEEZE condition removed as requested
    
    def get_multi_indicator_signal(self, df):
        """
        Get signals based on multi-indicator confirmation with stronger consolidated validation
        and enhanced filtering to reduce false positives
        """
        if len(df) < 10:  # Increased minimum data requirement for better context
            return None
            
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        prev3 = df.iloc[-3] if len(df) > 3 else None
        prev5 = df.iloc[-5] if len(df) > 5 else None
        
        # Count bullish and bearish signals with weighting for stronger indicators
        bullish_signals = 0
        bearish_signals = 0
        
        # Check for trend consistency over multiple candles (reduces false signals)
        is_consistent_trend = False
        if len(df) >= 5:
            # Check if the last 5 candles show consistent trend direction
            last_5_supertrend = [df.iloc[-i]['supertrend_direction'] for i in range(1, 6)]
            if all(direction == 1 for direction in last_5_supertrend):
                is_consistent_trend = True
                bullish_signals += 1  # Bonus for trend consistency
            elif all(direction == -1 for direction in last_5_supertrend):
                is_consistent_trend = True
                bearish_signals += 1  # Bonus for trend consistency
        
        # === PRIMARY INDICATORS (Higher weight) ===
        
        # Supertrend (stronger weight x2.5 for consistent trends)
        if latest['supertrend_direction'] == 1:
            bullish_signals += 2.5 if is_consistent_trend else 2.0
        else:
            bearish_signals += 2.5 if is_consistent_trend else 2.0
            
        # Price action relative to key levels with stronger confirmation
        if (latest['close'] > latest['ema_slow'] and 
            latest['close'] > latest['vwap'] and
            latest['close'] > latest['ema_fast']):  # Added requirement for all EMAs
            bullish_signals += 2.0  # Increased weight for stronger price action
        elif (latest['close'] < latest['ema_slow'] and 
              latest['close'] < latest['vwap'] and
              latest['close'] < latest['ema_fast']):  # Added requirement for all EMAs
            bearish_signals += 2.0  # Increased weight for stronger price action
        
        # Trend direction change (stronger weight)
        if prev['supertrend_direction'] == -1 and latest['supertrend_direction'] == 1:
            bullish_signals += 2  # Fresh bullish trend
        elif prev['supertrend_direction'] == 1 and latest['supertrend_direction'] == -1:
            bearish_signals += 2  # Fresh bearish trend
            
        # === SECONDARY INDICATORS ===
        
        # RSI & Volume-weighted RSI
        if latest['rsi'] < 30:
            bullish_signals += 1
        elif latest['rsi'] > 70:
            bearish_signals += 1
            
        # More extreme RSI values (stronger weight)
        if latest['rsi'] < 20:
            bullish_signals += 0.5  # Very oversold
        elif latest['rsi'] > 80:
            bearish_signals += 0.5  # Very overbought
            
        # Volume-weighted RSI
        if latest['volume_weighted_rsi'] < 25:
            bullish_signals += 1
        elif latest['volume_weighted_rsi'] > 75:
            bearish_signals += 1
            
        # MACD
        if latest['macd_crossover'] == 1:
            bullish_signals += 1
        elif latest['macd_crossover'] == -1:
            bearish_signals += 1
            
        # Volume confirmation
        if latest['volume_ratio'] > 1.5:
            # High volume adds weight to the dominant direction
            if bullish_signals > bearish_signals:
                bullish_signals += 1
            elif bearish_signals > bullish_signals:
                bearish_signals += 1
            
        # ADX trend strength
        if latest['adx'] > self.adx_threshold:
            if latest['di_plus'] > latest['di_minus']:
                bullish_signals += 1
            else:
                bearish_signals += 1
                
        # Strong ADX (higher weight)
        if latest['adx'] > self.adx_threshold * 1.5:
            if latest['di_plus'] > latest['di_minus']:
                bullish_signals += 0.5  # Strong trend confirmation
            else:
                bearish_signals += 0.5  # Strong trend confirmation
                
        # === SUPPORT/RESISTANCE CONFIRMATIONS ===
                
        # Price relative to VWAP
        if latest['close'] < latest['vwap'] * 0.98:
            bullish_signals += 0.5  # Potential support
        elif latest['close'] > latest['vwap'] * 1.02:
            bearish_signals += 0.5  # Potential resistance
            
        # Bollinger Bands
        if latest['close'] < latest['bb_lower'] * 1.01:
            bullish_signals += 1  # At support
        elif latest['close'] > latest['bb_upper'] * 0.99:
            bearish_signals += 1  # At resistance
        
        # Fibonacci levels (higher weight)
        close_to_fib_support = any(abs(latest['close'] - level) / latest['close'] < 0.005 for level in self.fib_support_levels)
        close_to_fib_resistance = any(abs(latest['close'] - level) / latest['close'] < 0.005 for level in self.fib_resistance_levels)
        
        if close_to_fib_support:
            bullish_signals += 1.5  # Strong support
        if close_to_fib_resistance:
            bearish_signals += 1.5  # Strong resistance
            
        # === MARKET CONDITION ADJUSTMENT ===
        
        # Adjust signal thresholds based on market condition - higher thresholds to reduce false signals
        market_condition = latest['market_condition']
        bull_threshold = 7.0  # Increased base threshold for stronger confirmation
        bear_threshold = 7.0  # Increased base threshold for stronger confirmation
        
        # Volume requirement - ensure we have enough volume to validate signals
        min_volume_ratio = 1.2  # Minimum volume needed relative to average
        has_sufficient_volume = latest['volume_ratio'] >= min_volume_ratio
        
        # Add requirement for sufficient volume in trending markets
        if not has_sufficient_volume:
            bull_threshold += 1.0  # Make it harder to trigger without enough volume
            bear_threshold += 1.0  # Make it harder to trigger without enough volume
        
        if market_condition in ['BULLISH', 'EXTREME_BULLISH']:
            bull_threshold -= 1.0  # Easier to trigger buy in bullish market, but still higher than before
            bear_threshold += 1.5  # Much harder to trigger sell in bullish market
        elif market_condition in ['BEARISH', 'EXTREME_BEARISH']:
            bull_threshold += 1.5  # Much harder to trigger buy in bearish market
            bear_threshold -= 1.0  # Easier to trigger sell in bearish market, but still higher than before
            
        # === FINAL SIGNAL DETERMINATION - ENHANCED FOR HIGHER WIN RATE ===
        
        # Calculate signal strength as a percentage of max possible
        max_possible_score = 20  # Base value remains 20
        bull_strength = bullish_signals / max_possible_score * 100
        bear_strength = bearish_signals / max_possible_score * 100
        
        # Check for trend continuation or reversal with stronger momentum requirements
        has_momentum = False
        has_strong_momentum = False
        if len(df) >= 5:  # Check more candles for stronger momentum confirmation
            # Look at the last 3 candles instead of just 2 for better momentum detection
            candle_momentums = [df.iloc[-i]['close'] - df.iloc[-i]['open'] for i in range(1, 4)]
            
            # If momentum is in the same direction for 3+ candles, it's much stronger
            if all(mom > 0 for mom in candle_momentums) and bullish_signals > bearish_signals:
                bullish_signals += 2.0  # Increased bonus for strong momentum continuation
                has_momentum = True
                has_strong_momentum = True
            elif all(mom < 0 for mom in candle_momentums) and bearish_signals > bullish_signals:
                bearish_signals += 2.0  # Increased bonus for strong momentum continuation
                has_momentum = True
                has_strong_momentum = True
            # Check for 2 candles momentum (weaker)
            elif (candle_momentums[0] > 0 and candle_momentums[1] > 0 and bullish_signals > bearish_signals):
                bullish_signals += 1.0  # Standard bonus for momentum continuation
                has_momentum = True
            elif (candle_momentums[0] < 0 and candle_momentums[1] < 0 and bearish_signals > bullish_signals):
                bearish_signals += 1.0  # Standard bonus for momentum continuation
                has_momentum = True
        
        logger.debug(f"Multi-indicator signals - Bullish: {bullish_signals:.1f} ({bull_strength:.1f}%), " +
                    f"Bearish: {bearish_signals:.1f} ({bear_strength:.1f}%)")
        
        # Add volume filter for extra quality confirmation
        has_volume_confirmation = latest['volume_ratio'] > 1.3  # Volume must be at least 30% above average
        
        # Return signal based on extremely strong confirmation from multiple indicators
        # With much more stringent requirements to greatly improve win rate
        if (bullish_signals >= bull_threshold and 
            bullish_signals > bearish_signals * 2.5 and  # Increased from 2.0x to 2.5x difference
            ((has_strong_momentum and bull_strength > 50) or  # Strong momentum with decent strength
             (has_momentum and bull_strength > 65) or  # Regular momentum with higher strength
             (has_volume_confirmation and bull_strength > 70))):  # No momentum but very high strength with volume
            logger.info(f"Strong bullish confirmation: {bullish_signals:.1f} signals ({bull_strength:.1f}%)")
            return 'BUY'
            
        if (bearish_signals >= bear_threshold and 
            bearish_signals > bullish_signals * 2.5 and  # Increased from 2.0x to 2.5x difference
            ((has_strong_momentum and bear_strength > 50) or  # Strong momentum with decent strength
             (has_momentum and bear_strength > 65) or  # Regular momentum with higher strength
             (has_volume_confirmation and bear_strength > 70))):  # No momentum but very high strength with volume
            logger.info(f"Strong bearish confirmation: {bearish_signals:.1f} signals ({bear_strength:.1f}%)")
            return 'SELL'
            
        return None
    
    def get_grid_signal(self, df):
        """Enhanced grid signal with position sizing - REVERSED SIGNALS"""
        latest = df.iloc[-1]
        current_price = latest['close']
        current_time = latest['open_time']
        
        # Check for cool-off period after consecutive losses
        if self.in_cooloff_period(current_time):
            logger.info(f"In cool-off period after loss. No grid signals until cool-off completes.")
            return None
        
        # If no grids, generate them first
        if self.grids is None or len(self.grids) == 0 or self.should_update_grids(df):
            self.grids = self.generate_grids(df)
            self.current_trend = latest['trend']
            self.current_market_condition = latest['market_condition']
            self.last_grid_update = latest['open_time']
            logger.info(f"Generated new grids for {self.current_market_condition} market condition")
            return None  # No signal on grid initialization
        
        # Find the nearest grid levels
        buy_grids = [grid for grid in self.grids if grid['type'] == 'BUY' and grid['status'] == 'ACTIVE']
        sell_grids = [grid for grid in self.grids if grid['type'] == 'SELL' and grid['status'] == 'ACTIVE']
        
        # Find closest buy and sell grids
        closest_buy = None
        closest_sell = None
        
        if buy_grids:
            closest_buy = max(buy_grids, key=lambda x: x['price'])
            
        if sell_grids:
            closest_sell = min(sell_grids, key=lambda x: x['price'])
        
        # Determine signal based on price position relative to grids
        # REVERSED: BUY grid triggers SELL signal instead and vice versa
        if closest_buy and current_price <= closest_buy['price'] * 1.001:
            # Mark this grid as triggered
            for grid in self.grids:
                if grid['price'] == closest_buy['price']:
                    grid['status'] = 'TRIGGERED'
                    
            # Update position size for risk manager
            if self.risk_manager:
                position_size = self.calculate_dynamic_position_size(df)
                self.risk_manager.update_position_sizing(position_size)
                self.position_size_pct = position_size
            
            # REVERSED: Return SELL instead of BUY
            return 'SELL'
            
        elif closest_sell and current_price >= closest_sell['price'] * 0.999:
            # Mark this grid as triggered
            for grid in self.grids:
                if grid['price'] == closest_sell['price']:
                    grid['status'] = 'TRIGGERED'
                    
            # Update position size for risk manager
            if self.risk_manager:
                position_size = self.calculate_dynamic_position_size(df)
                self.risk_manager.update_position_sizing(position_size)
                self.position_size_pct = position_size
            
            # REVERSED: Return BUY instead of SELL
            return 'BUY'
            
        return None
    
    def get_sideways_signal(self, df):
        """
        Enhanced sideways market signal with multiple confirmation factors for higher win rate
        WITH REVERSED SIGNALS
        """
        if len(df) < 5:  # Require more data for better analysis
            return None
            
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Calculate price distance to bands relative to total band width
        bb_total_width = latest['bb_upper'] - latest['bb_lower']
        if bb_total_width == 0:  # Avoid division by zero
            return None
            
        # Relative position within bands (0 = lower band, 1 = upper band)
        rel_position = (latest['close'] - latest['bb_lower']) / bb_total_width
        
        # Check for consecutive touches of bands (better confirmation)
        lower_band_touch_count = 0
        upper_band_touch_count = 0
        for i in range(1, min(4, len(df))):
            if df.iloc[-i]['low'] <= df.iloc[-i]['bb_lower'] * 1.01:
                lower_band_touch_count += 1
            if df.iloc[-i]['high'] >= df.iloc[-i]['bb_upper'] * 0.99:
                upper_band_touch_count += 1
        
        # Volume confirmation - higher volume on band tests is more reliable
        has_volume_spike = latest['volume_ratio'] > 1.4
        
        # ADX confirmation - ensure we're actually in a sideways market
        sideways_confirmed = latest['adx'] < 23
        
        # In sideways markets, use VWAP and key levels for higher probability setups
        
        # REVERSED: Buy near lower Bollinger Band with multiple confirmations becomes SELL
        if (rel_position < 0.15 and  # Very close to lower band (bottom 15%)
            latest['close'] < latest['vwap'] * 0.99 and  # Below VWAP with buffer
            lower_band_touch_count >= 2 and  # Multiple touches of lower band
            sideways_confirmed and
            has_volume_spike):  # Good volume on reversal
            logger.info(f"High-quality lower band touch in sideways market: BB position={rel_position:.2f}, Volume={latest['volume_ratio']:.1f}x")
            return 'SELL'  # REVERSED
            
        # REVERSED: Sell near upper Bollinger Band with multiple confirmations becomes BUY
        elif (rel_position > 0.85 and  # Very close to upper band (top 15%)
              latest['close'] > latest['vwap'] * 1.01 and  # Above VWAP with buffer
              upper_band_touch_count >= 2 and  # Multiple touches of upper band
              sideways_confirmed and
              has_volume_spike):  # Good volume on reversal
            logger.info(f"High-quality upper band touch in sideways market: BB position={rel_position:.2f}, Volume={latest['volume_ratio']:.1f}x")
            return 'BUY'  # REVERSED
            
        # REVERSED: Volume-weighted RSI signals with extra confirmation in sideways markets
        elif (latest['volume_weighted_rsi'] < 25 and  # More extreme threshold
              prev['volume_weighted_rsi'] < latest['volume_weighted_rsi'] and  # RSI turning up (bottoming)
              sideways_confirmed and
              latest['close'] < latest['bb_lower'] * 1.05):  # Close to lower band
            logger.info(f"High-quality oversold signal in sideways market: V-RSI={latest['volume_weighted_rsi']:.1f}")
            return 'SELL'  # REVERSED
            
        elif (latest['volume_weighted_rsi'] > 75 and  # More extreme threshold
              prev['volume_weighted_rsi'] > latest['volume_weighted_rsi'] and  # RSI turning down (topping)
              sideways_confirmed and
              latest['close'] > latest['bb_upper'] * 0.95):  # Close to upper band
            logger.info(f"High-quality overbought signal in sideways market: V-RSI={latest['volume_weighted_rsi']:.1f}")
            return 'BUY'  # REVERSED
            
        return None
    
    def get_bullish_signal(self, df):
        """
        Significantly enhanced signal for bullish market with strong confirming conditions
        to dramatically improve win rate
        WITH REVERSED SIGNALS
        """
        if len(df) < 5:  # Increased minimum data requirement
            return None
            
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            prev2 = df.iloc[-3] if len(df) > 3 else None
            prev3 = df.iloc[-4] if len(df) > 4 else None
            market_condition = latest['market_condition']
            
            # Adjust RSI thresholds based on market condition
            rsi_oversold = 22 if market_condition == 'EXTREME_BULLISH' else 30
            
            # Check for multiple candle confirmation (higher win rate signal)
            bullish_candles = 0
            for i in range(1, min(4, len(df))):
                if df.iloc[-i]['close'] > df.iloc[-i]['open']:
                    bullish_candles += 1
                    
            bearish_candles = 0
            for i in range(1, min(4, len(df))):
                if df.iloc[-i]['close'] < df.iloc[-i]['open']:
                    bearish_candles += 1
            
            # REVERSED: More selective and higher quality oversold conditions for SELL signals in bullish markets
            if (latest['rsi'] < rsi_oversold and 
                latest['volume_ratio'] > 1.4 and  # Stronger volume requirement
                latest['close'] < latest['bb_lower'] * 1.01 and  # Price near/below lower BB
                bearish_candles >= 2):  # Multiple bearish candles confirmation
                logger.info(f"High-quality oversold signal in bullish market: RSI={latest['rsi']:.1f}, Volume={latest['volume_ratio']:.1f}x")
                return 'SELL'  # REVERSED
                
            # REVERSED: SELL on MACD crossover with multiple confirmation factors
            if (prev['macd'] < prev['macd_signal'] and 
                latest['macd'] > latest['macd_signal'] and 
                latest['volume_ratio'] > 1.5 and  # Much stronger volume confirmation
                latest['supertrend_direction'] == 1 and  # Aligned with supertrend
                latest['close'] > latest['ema_fast']):  # Price above fast EMA
                logger.info(f"High-quality MACD crossover in bullish market with volume={latest['volume_ratio']:.1f}x")
                return 'SELL'  # REVERSED
                
            # REVERSED: SELL on Supertrend direction change with confirmation
            if (prev['supertrend_direction'] == -1 and 
                latest['supertrend_direction'] == 1 and
                latest['volume_ratio'] > 1.3 and  # Volume confirmation
                latest['adx'] > 25 and  # Decent trend strength
                latest['di_plus'] > latest['di_minus']):  # Positive direction
                logger.info(f"High-quality Supertrend reversal in bullish market: ADX={latest['adx']:.1f}")
                return 'SELL'  # REVERSED
                
            # REVERSED: Buy only on extreme overbought conditions with multiple confirmations
            if (latest['rsi'] > 78 and 
                latest['close'] > latest['bb_upper'] * 1.02 and  # More extreme extension
                latest['close'] > latest['vwap'] * 1.05 and  # Further above VWAP
                bearish_candles >= 2 and  # Multiple bearish candles
                latest['volume_ratio'] > 1.5):  # Higher volume
                logger.info(f"High-quality overbought signal in bullish market: RSI={latest['rsi']:.1f}")
                return 'BUY'  # REVERSED
                
            return None
            
        except Exception as e:
            logger.error(f"Error in get_bullish_signal: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Error in get_bullish_signal: {e}")
            return None
    
    def get_bearish_signal(self, df):
        """
        Significantly enhanced signal for bearish market with strong confirming conditions
        to dramatically improve win rate
        WITH REVERSED SIGNALS
        """
        if len(df) < 5:  # Increased minimum data requirement
            return None
            
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            prev2 = df.iloc[-3] if len(df) > 3 else None
            prev3 = df.iloc[-4] if len(df) > 4 else None
            market_condition = latest['market_condition']
            
            # Adjust RSI thresholds based on market condition
            rsi_overbought = 78 if market_condition == 'EXTREME_BEARISH' else 70
            
            # Check for multiple candle confirmation (higher win rate signal)
            bullish_candles = 0
            for i in range(1, min(4, len(df))):
                if df.iloc[-i]['close'] > df.iloc[-i]['open']:
                    bullish_candles += 1
                    
            bearish_candles = 0
            for i in range(1, min(4, len(df))):
                if df.iloc[-i]['close'] < df.iloc[-i]['open']:
                    bearish_candles += 1
            
            # REVERSED: More selective and higher quality overbought conditions for BUY signals in bearish markets
            if (latest['rsi'] > rsi_overbought and 
                latest['volume_ratio'] > 1.4 and  # Stronger volume requirement
                latest['close'] > latest['bb_upper'] * 0.99 and  # Price near/above upper BB
                bullish_candles >= 2):  # Multiple bullish candles confirmation
                logger.info(f"High-quality overbought signal in bearish market: RSI={latest['rsi']:.1f}, Volume={latest['volume_ratio']:.1f}x")
                return 'BUY'  # REVERSED
                
            # REVERSED: BUY on MACD crossover with multiple confirmation factors
            if (prev['macd'] > prev['macd_signal'] and 
                latest['macd'] < latest['macd_signal'] and 
                latest['volume_ratio'] > 1.5 and  # Much stronger volume confirmation
                latest['supertrend_direction'] == -1 and  # Aligned with supertrend
                latest['close'] < latest['ema_fast']):  # Price below fast EMA
                logger.info(f"High-quality MACD crossover in bearish market with volume={latest['volume_ratio']:.1f}x")
                return 'BUY'  # REVERSED
                
            # REVERSED: BUY on Supertrend direction change with confirmation
            if (prev['supertrend_direction'] == 1 and 
                latest['supertrend_direction'] == -1 and
                latest['volume_ratio'] > 1.3 and  # Volume confirmation
                latest['adx'] > 25 and  # Decent trend strength
                latest['di_minus'] > latest['di_plus']):  # Negative direction
                logger.info(f"High-quality Supertrend reversal in bearish market: ADX={latest['adx']:.1f}")
                return 'BUY'  # REVERSED
                
            # REVERSED: SELL only on extreme oversold conditions with multiple confirmations
            if (latest['rsi'] < 22 and 
                latest['close'] < latest['bb_lower'] * 0.98 and  # More extreme extension
                latest['close'] < latest['vwap'] * 0.95 and  # Further below VWAP
                bullish_candles >= 2 and  # Multiple bullish candles
                latest['volume_ratio'] > 1.5):  # Higher volume
                logger.info(f"High-quality oversold signal in bearish market: RSI={latest['rsi']:.1f}")
                return 'SELL'  # REVERSED
                
            return None
            
        except Exception as e:
            logger.error(f"Error in get_bearish_signal: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Error in get_bearish_signal: {e}")
            return None
            
    def update_trade_result(self, was_profitable):
        """
        Update consecutive losses counter for cool-off period calculation
        
        Args:
            was_profitable: Boolean indicating if the last trade was profitable
        """
        if was_profitable:
            # Reset consecutive losses on profitable trade
            self.consecutive_losses = 0
            self.last_loss_time = None
            logger.info("Profitable trade - reset consecutive losses counter")

        else:
            # Increment consecutive losses
            self.consecutive_losses += 1
            self.last_loss_time = datetime.now()
            logger.info(f"Loss recorded - consecutive losses: {self.consecutive_losses}")
            
            # Modified to enter cool-off period after just 1 loss
            logger.info(f"Entering cool-off period for {self.cooloff_period} candles after experiencing a loss")
            
            # Send Telegram notification about entering cool-off
            try:
                from modules.telegram_notifier import TelegramNotifier
                notifier = TelegramNotifier()
                notifier.send_message(f"⚠️ *Cool-off Period Started*\n\nBot has experienced a loss and is entering cool-off mode for {self.cooloff_period} minutes. Trading will be paused during this period.")
            except Exception as e:
                logger.error(f"Failed to send Telegram notification: {e}")
    
    def get_extreme_market_signal(self, df):
        """
        Specialized signal generation for extreme market conditions
        WITH REVERSED SIGNALS
        """
        if len(df) < 3:
            return None
            
        latest = df.iloc[-1]
        market_condition = latest['market_condition']
        
        # Only process if we're in extreme market conditions
        if market_condition not in ['EXTREME_BULLISH', 'EXTREME_BEARISH']:
            return None
            
        # In extreme bullish market
        if market_condition == 'EXTREME_BULLISH':
            # REVERSED: Look for SELLING opportunities on small dips (was buying)
            if (latest['close'] < latest['vwap'] and 
                latest['supertrend_direction'] == 1 and
                latest['rsi'] < 40):
                return 'SELL'  # REVERSED
                
        # In extreme bearish market
        elif market_condition == 'EXTREME_BEARISH':
            # REVERSED: Look for BUYING opportunities on small rallies (was selling)
            if (latest['close'] > latest['vwap'] and 
                latest['supertrend_direction'] == -1 and
                latest['rsi'] > 60):
                return 'BUY'  # REVERSED
                
        return None
    
    def get_signal(self, klines):
        """
        Enhanced signal generation integrating all the new features
        WITH SIGNALS REVERSED (BUY → SELL and SELL → BUY)
        """
        # Prepare and add indicators to the data
        df = self.prepare_data(klines)
        df = self.add_indicators(df)
        
        if len(df) < self.trend_ema_slow + 5:
            # Not enough data to generate reliable signals
            return None
        
        # Get latest data
        latest = df.iloc[-1]
        market_condition = latest['market_condition']
        current_time = latest['open_time']
        
        # Update risk manager with current market condition
        if self.risk_manager:
            self.risk_manager.set_market_condition(market_condition)
        
        # Check for cool-off period first
        if self.in_cooloff_period(current_time):
            logger.info(f"In cool-off period after loss. No trading signals until cool-off completes.")
            return None
        
        # 1. Check for V-shaped reversals in extreme market conditions
        reversal_signal = self.get_v_reversal_signal(df)
        if reversal_signal:
            logger.info(f"V-reversal detected in {market_condition} market. Signal: {reversal_signal}")
            # REVERSED: Return the opposite signal
            return "SELL" if reversal_signal == "BUY" else "BUY"
        
        # 2. Removed SQUEEZE condition check as requested
        
        # 3. Check for multi-indicator confirmation signals
        multi_signal = self.get_multi_indicator_signal(df)
        if multi_signal:
            logger.info(f"Multi-indicator confirmation. Signal: {multi_signal}")
            # REVERSED: Return the opposite signal
            return "SELL" if multi_signal == "BUY" else "BUY"
        
        # 4. Get grid signal (works in all market conditions)
        grid_signal = self.get_grid_signal(df)
        # REVERSED: If we have a grid signal, reverse it
        grid_signal = "SELL" if grid_signal == "BUY" else ("BUY" if grid_signal == "SELL" else None)
        
        # 5. Get specific signals based on market condition
        if market_condition in ['EXTREME_BULLISH', 'EXTREME_BEARISH']:
            condition_signal = self.get_extreme_market_signal(df)
            logger.debug(f"EXTREME market detected. Grid signal: {grid_signal}, Extreme signal: {condition_signal}")
            
            # In extreme market conditions, prefer the condition-specific signal
            if condition_signal:
                # REVERSED: Return the opposite signal
                return "SELL" if condition_signal == "BUY" else "BUY"
                
        elif market_condition in ['BULLISH', 'BEARISH']:
            if market_condition == 'BULLISH':
                condition_signal = self.get_bullish_signal(df)
            else:
                condition_signal = self.get_bearish_signal(df)
                
            logger.debug(f"{market_condition} market detected. Grid signal: {grid_signal}, Condition signal: {condition_signal}")
            
            # In trending markets, prefer the trending signal
            if condition_signal:
                # REVERSED: Return the opposite signal
                return "SELL" if condition_signal == "BUY" else "BUY"
                
        elif market_condition == 'SIDEWAYS':
            condition_signal = self.get_sideways_signal(df)
            logger.debug(f"SIDEWAYS market detected. Grid signal: {grid_signal}, Sideways signal: {condition_signal}")
            
            # In sideways markets, prioritize mean reversion signals
            if condition_signal:
                # REVERSED: Return the opposite signal
                return "SELL" if condition_signal == "BUY" else "BUY"
                
        # 6. Default to grid signal if no specialized signal was returned
        return grid_signal


# Update the factory function to include only SUI strategy
def get_strategy(strategy_name):
    """Factory function to get a strategy by name"""
    from modules.config import (
        # SUI parameters
        SUI_GRID_LEVELS, SUI_GRID_SPACING_PCT, SUI_TREND_EMA_FAST, SUI_TREND_EMA_SLOW,
        SUI_VOLATILITY_LOOKBACK, RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD,
        SUI_VOLUME_MA_PERIOD, SUI_ADX_PERIOD, SUI_ADX_THRESHOLD, SUI_SIDEWAYS_THRESHOLD,
        SUI_VOLATILITY_MULTIPLIER, SUI_TREND_CONDITION_MULTIPLIER,
        SUI_MIN_GRID_SPACING, SUI_MAX_GRID_SPACING
    )
    
    strategies = {
        'SuiDynamicGridStrategy': SuiDynamicGridStrategy(
            grid_levels=SUI_GRID_LEVELS,
            grid_spacing_pct=SUI_GRID_SPACING_PCT,
            trend_ema_fast=SUI_TREND_EMA_FAST,
            trend_ema_slow=SUI_TREND_EMA_SLOW,
            volatility_lookback=SUI_VOLATILITY_LOOKBACK,
            rsi_period=RSI_PERIOD,
            rsi_overbought=RSI_OVERBOUGHT,
            rsi_oversold=RSI_OVERSOLD,
            volume_ma_period=SUI_VOLUME_MA_PERIOD,
            adx_period=SUI_ADX_PERIOD,
            adx_threshold=SUI_ADX_THRESHOLD,
            sideways_threshold=SUI_SIDEWAYS_THRESHOLD,
            # Pass SUI specific parameters
            volatility_multiplier=SUI_VOLATILITY_MULTIPLIER,
            trend_condition_multiplier=SUI_TREND_CONDITION_MULTIPLIER,
            min_grid_spacing=SUI_MIN_GRID_SPACING,
            max_grid_spacing=SUI_MAX_GRID_SPACING
        )
    }
    
    if strategy_name in strategies:
        return strategies[strategy_name]
    
    logger.warning(f"Strategy {strategy_name} not found. Defaulting to base trading strategy.")
    return TradingStrategy(strategy_name)


def get_strategy_for_symbol(symbol, strategy_name=None):
    """Get the appropriate strategy based on the trading symbol"""
    # If a specific strategy is requested, use it
    if strategy_name:
        return get_strategy(strategy_name)
    
    # Default to SUIUSDT strategy for any symbol
    return SuiDynamicGridStrategy()
    
    # Default to base strategy if needed
    # return TradingStrategy(symbol)

# End of file