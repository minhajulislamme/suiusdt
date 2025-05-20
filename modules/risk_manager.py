import logging
import math
import pandas as pd
import ta
from modules.config import (
    INITIAL_BALANCE, RISK_PER_TRADE, MAX_OPEN_POSITIONS,
    USE_STOP_LOSS, STOP_LOSS_PCT, USE_TAKE_PROFIT, 
    TAKE_PROFIT_PCT, TRAILING_TAKE_PROFIT, TRAILING_TAKE_PROFIT_PCT, TRAILING_STOP, TRAILING_STOP_PCT,
    AUTO_COMPOUND, COMPOUND_REINVEST_PERCENT,
    # Adaptive risk management settings
    STOP_LOSS_PCT_BULLISH, STOP_LOSS_PCT_BEARISH, STOP_LOSS_PCT_SIDEWAYS,
    TAKE_PROFIT_PCT_BULLISH, TAKE_PROFIT_PCT_BEARISH, TAKE_PROFIT_PCT_SIDEWAYS,
    TRAILING_STOP_PCT_BULLISH, TRAILING_STOP_PCT_BEARISH, TRAILING_STOP_PCT_SIDEWAYS,
    TRAILING_TAKE_PROFIT_PCT_BULLISH, TRAILING_TAKE_PROFIT_PCT_BEARISH, TRAILING_TAKE_PROFIT_PCT_SIDEWAYS,
    # Multi-instance mode settings
    MULTI_INSTANCE_MODE, MAX_POSITIONS_PER_SYMBOL
)

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, binance_client):
        """Initialize risk manager with a reference to binance client"""
        self.binance_client = binance_client
        self.initial_balance = None
        self.last_known_balance = None
        self.current_market_condition = None  # Will be set to 'BULLISH', 'BEARISH', or 'SIDEWAYS'
        self.position_size_multiplier = 1.0  # Default position size multiplier
        
    def set_market_condition(self, market_condition):
        """Set the current market condition for adaptive risk management"""
        if market_condition in ['BULLISH', 'BEARISH', 'SIDEWAYS', 'EXTREME_BULLISH', 'EXTREME_BEARISH']:
            if self.current_market_condition != market_condition:
                logger.info(f"Market condition changed to {market_condition}")
                self.current_market_condition = market_condition
        else:
            logger.warning(f"Invalid market condition: {market_condition}. Using default risk parameters.")
    
    def update_position_sizing(self, position_size_multiplier):
        """
        Update the position size multiplier based on market conditions and volatility
        
        Args:
            position_size_multiplier: A multiplier to adjust position size (0.5 = 50%, 1.0 = 100%, etc.)
        """
        if position_size_multiplier <= 0:
            logger.warning(f"Invalid position size multiplier: {position_size_multiplier}. Using default value of 1.0")
            position_size_multiplier = 1.0
            
        self.position_size_multiplier = position_size_multiplier
        logger.info(f"Position size multiplier updated to {position_size_multiplier:.2f}")
        
    def calculate_position_size(self, symbol, side, price, stop_loss_price=None):
        """
        Calculate position size based on risk parameters
        
        Args:
            symbol: Trading pair symbol
            side: 'BUY' or 'SELL'
            price: Current market price
            stop_loss_price: Optional stop loss price for calculating risk
            
        Returns:
            quantity: The position size
        """
        # Get account balance
        balance = self.binance_client.get_account_balance()
        
        # Initialize initial balance if not set
        if self.initial_balance is None:
            self.initial_balance = balance
            self.last_known_balance = balance
            
        # Auto compound logic
        if AUTO_COMPOUND and self.last_known_balance is not None:
            profit = balance - self.last_known_balance
            if profit > 0:
                # We've made profit, apply compounding by increasing risk amount
                logger.info(f"Auto-compounding profit of {profit:.2f} USDT")
                # Update the last known balance
                self.last_known_balance = balance
            
        if balance <= 0:
            logger.error("Insufficient balance to open a position")
            return 0
            
        # Get symbol info for precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Could not retrieve symbol info for {symbol}")
            return 0
            
        # Calculate risk amount - use higher risk for small accounts
        small_account = balance < 100.0  # Consider accounts under $100 as small
        
        # Adjust risk per trade for small accounts - more aggressive but safer than no trades
        effective_risk = RISK_PER_TRADE
        if small_account:
            effective_risk = max(RISK_PER_TRADE, 0.05)  # Minimum 5% risk for small accounts
            logger.info(f"Small account detected (${balance:.2f}). Using {effective_risk*100:.1f}% risk per trade.")
            
        risk_amount = balance * effective_risk
        
        # Calculate position size based on risk and stop loss
        if stop_loss_price and USE_STOP_LOSS:
            # If stop loss is provided, calculate size based on it
            risk_per_unit = abs(price - stop_loss_price)
            if risk_per_unit <= 0:
                logger.error("Stop loss too close to entry price")
                return 0
                
            # Calculate max quantity based on risk
            max_quantity = risk_amount / risk_per_unit
        else:
            # If no stop loss, use a percentage of balance with leverage
            leverage = self.get_current_leverage(symbol)
            max_quantity = (balance * effective_risk * leverage) / price
        
        # Apply precision to quantity
        quantity_precision = symbol_info['quantity_precision']
        quantity = round_step_size(max_quantity, get_step_size(symbol_info['min_qty']))
        
        # Check minimum notional
        min_notional = symbol_info['min_notional']
        if quantity * price < min_notional:
            logger.warning(f"Position size too small - below minimum notional of {min_notional}")
            
            # For small accounts, force minimum notional even if it exceeds normal risk parameters
            if small_account:
                min_quantity = math.ceil(min_notional / price * 10**quantity_precision) / 10**quantity_precision
                
                # Make sure we don't use more than 50% of balance for very small accounts
                max_safe_quantity = (balance * 0.5 * leverage) / price
                max_safe_quantity = math.floor(max_safe_quantity * 10**quantity_precision) / 10**quantity_precision
                
                quantity = min(min_quantity, max_safe_quantity)
                
                if quantity * price / leverage > balance * 0.5:
                    logger.warning("Position would use more than 50% of balance - reducing size")
                    quantity = math.floor((balance * 0.5 * leverage / price) * 10**quantity_precision) / 10**quantity_precision
                
                if quantity > 0:
                    logger.info(f"Small account: Adjusted position size to meet minimum notional: {quantity}")
                else:
                    logger.error("Balance too low to open even minimum position")
                    return 0
            else:
                # Normal account handling
                if min_notional / price <= max_quantity:
                    quantity = math.ceil(min_notional / price * 10**quantity_precision) / 10**quantity_precision
                    logger.info(f"Adjusted position size to meet minimum notional: {quantity}")
                else:
                    logger.error(f"Cannot meet minimum notional with current risk settings")
                    return 0
                
        logger.info(f"Calculated position size: {quantity} units at {price} per unit")
        return quantity
        
    def get_current_leverage(self, symbol):
        """Get the current leverage for a symbol"""
        position_info = self.binance_client.get_position_info(symbol)
        if position_info:
            return position_info['leverage']
        return 1  # Default to 1x if no position info
        
    def should_open_position(self, symbol):
        """Check if a new position should be opened based on risk rules"""
        # Check if we already have an open position for this symbol
        position_info = self.binance_client.get_position_info(symbol)
        if position_info and abs(position_info['position_amount']) > 0:
            logger.info(f"Already have an open position for {symbol}")
            return False
            
        # Check maximum number of open positions (only for the current trading symbol)
        # This allows separate bot instances for different trading pairs to operate independently
        if MULTI_INSTANCE_MODE:
            # In multi-instance mode, only count positions for the current symbol
            positions = self.binance_client.client.futures_position_information()
            # Check if we've reached the max positions for this symbol
            symbol_positions = [p for p in positions if p['symbol'] == symbol and float(p['positionAmt']) != 0]
            if len(symbol_positions) >= MAX_POSITIONS_PER_SYMBOL:
                logger.info(f"Maximum number of positions for {symbol} ({MAX_POSITIONS_PER_SYMBOL}) reached")
                return False
        else:
            # Original behavior - count all positions
            positions = self.binance_client.client.futures_position_information()
            open_positions = [p for p in positions if float(p['positionAmt']) != 0]
            if len(open_positions) >= MAX_OPEN_POSITIONS:
                logger.info(f"Maximum number of open positions ({MAX_OPEN_POSITIONS}) reached")
                return False
            
        return True
        
    def calculate_stop_loss(self, symbol, side, entry_price):
        """Calculate stop loss price based on configuration and market condition"""
        if not USE_STOP_LOSS:
            return None
            
        # Special handling for SUI tokens which have higher volatility
        is_sui = 'SUI' in symbol
            
        # Choose stop loss percentage based on market condition
        if self.current_market_condition == 'BULLISH':
            stop_loss_pct = STOP_LOSS_PCT_BULLISH
        elif self.current_market_condition == 'BEARISH':
            stop_loss_pct = STOP_LOSS_PCT_BEARISH
        elif self.current_market_condition == 'SIDEWAYS':
            stop_loss_pct = STOP_LOSS_PCT_SIDEWAYS
        else:
            stop_loss_pct = STOP_LOSS_PCT  # Default            # For SUI tokens, add more buffer to the stop loss to account for higher volatility
            if is_sui:
                original_pct = stop_loss_pct
                # Increased multiplier from 1.7 to 2.3 (130% wider) to avoid premature stop losses
                stop_loss_pct = stop_loss_pct * 2.3
                logger.info(f"SUI token detected: Increasing stop loss percentage from {original_pct*100:.2f}% to {stop_loss_pct*100:.2f}%")
            
        if side == "BUY":  # Long position
            stop_price = entry_price * (1 - stop_loss_pct)
        else:  # Short position
            stop_price = entry_price * (1 + stop_loss_pct)
            
        # Apply price precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if symbol_info:
            price_precision = symbol_info['price_precision']
            stop_price = round(stop_price, price_precision)
            
        if is_sui:
            logger.info(f"Calculated SUI-specific {self.current_market_condition} stop loss at {stop_price} ({stop_loss_pct*100:.2f}%, enhanced buffer active)")
        else:
            logger.info(f"Calculated {self.current_market_condition} stop loss at {stop_price} ({stop_loss_pct*100}%)")
        return stop_price
        
    def calculate_take_profit(self, symbol, side, entry_price):
        """Calculate take profit price based on configuration and market condition"""
        if not USE_TAKE_PROFIT:
            return None
            
        # Choose take profit percentage based on market condition
        if self.current_market_condition == 'BULLISH':
            take_profit_pct = TAKE_PROFIT_PCT_BULLISH
        elif self.current_market_condition == 'BEARISH':
            take_profit_pct = TAKE_PROFIT_PCT_BEARISH
        elif self.current_market_condition == 'SIDEWAYS':
            take_profit_pct = TAKE_PROFIT_PCT_SIDEWAYS
        else:
            take_profit_pct = TAKE_PROFIT_PCT  # Default
            
        if side == "BUY":  # Long position
            take_profit_price = entry_price * (1 + take_profit_pct)
        else:  # Short position
            take_profit_price = entry_price * (1 - take_profit_pct)
            
        # Apply price precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if symbol_info:
            price_precision = symbol_info['price_precision']
            take_profit_price = round(take_profit_price, price_precision)
            
        logger.info(f"Calculated {self.current_market_condition} take profit at {take_profit_price} ({take_profit_pct*100}%)")
        return take_profit_price
        
    def adjust_stop_loss_for_trailing(self, symbol, side, current_price, position_info=None):
        """Adjust stop loss for trailing stop if needed"""
        if not TRAILING_STOP:
            return None
            
        if not position_info:
            # Get position info specifically for this symbol (important for multi-instance mode)
            position_info = self.binance_client.get_position_info(symbol)
            
        # Only proceed if we have a valid position for this specific symbol
        if not position_info or abs(position_info['position_amount']) == 0:
            return None
            
        # Ensure we're dealing with the right symbol in multi-instance mode
        if position_info['symbol'] != symbol:
            logger.warning(f"Position symbol mismatch: expected {symbol}, got {position_info['symbol']}")
            return None
            
        entry_price = position_info['entry_price']
        
        # Choose trailing stop percentage based on market condition
        if self.current_market_condition == 'BULLISH':
            trailing_stop_pct = TRAILING_STOP_PCT_BULLISH
        elif self.current_market_condition == 'BEARISH':
            trailing_stop_pct = TRAILING_STOP_PCT_BEARISH
        elif self.current_market_condition == 'SIDEWAYS':
            trailing_stop_pct = TRAILING_STOP_PCT_SIDEWAYS
        else:
            trailing_stop_pct = TRAILING_STOP_PCT  # Default
        
        # Calculate new stop loss based on current price
        if side == "BUY":  # Long position
            new_stop = current_price * (1 - trailing_stop_pct)
            # Only move stop loss up, never down
            current_stop = self.calculate_stop_loss(symbol, side, entry_price)
            if current_stop and new_stop <= current_stop:
                logger.debug(f"Not adjusting trailing stop for long position: current ({current_stop}) > calculated ({new_stop})")
                return None
        else:  # Short position
            new_stop = current_price * (1 + trailing_stop_pct)
            # Only move stop loss down, never up
            current_stop = self.calculate_stop_loss(symbol, side, entry_price)
            if current_stop and new_stop >= current_stop:
                logger.debug(f"Not adjusting trailing stop for short position: current ({current_stop}) < calculated ({new_stop})")
                return None
                
        # Apply price precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if symbol_info:
            price_precision = symbol_info['price_precision']
            new_stop = round(new_stop, price_precision)
            
        logger.info(f"Adjusted {self.current_market_condition} trailing stop loss to {new_stop} ({trailing_stop_pct*100}%)")
        logger.info(f"Current price: {current_price}, Entry price: {entry_price}, Stop loss moved: {current_stop} -> {new_stop}")
        return new_stop
        
    def adjust_take_profit_for_trailing(self, symbol, side, current_price, position_info=None):
        """
        Adjust take profit price based on trailing settings
        
        Args:
            symbol: Trading pair symbol
            side: Position side ('BUY' or 'SELL')
            current_price: Current market price
            position_info: Position information including entry_price
            
        Returns:
            new_take_profit: New take profit price if it should be adjusted, None otherwise
        """
        if not USE_TAKE_PROFIT or not TRAILING_TAKE_PROFIT:
            return None
            
        if not position_info:
            return None
            
        entry_price = float(position_info.get('entry_price', 0))
        if entry_price <= 0:
            return None
        
        # Get symbol info for precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if not symbol_info:
            return None
            
        price_precision = symbol_info.get('price_precision', 2)
        
        # Choose trailing take profit percentage based on market condition
        if self.current_market_condition == 'BULLISH':
            trailing_take_profit_pct = TRAILING_TAKE_PROFIT_PCT_BULLISH
        elif self.current_market_condition == 'BEARISH':
            trailing_take_profit_pct = TRAILING_TAKE_PROFIT_PCT_BEARISH
        elif self.current_market_condition == 'SIDEWAYS':
            trailing_take_profit_pct = TRAILING_TAKE_PROFIT_PCT_SIDEWAYS
        else:
            trailing_take_profit_pct = TRAILING_TAKE_PROFIT_PCT  # Default
        
        # Calculate the current dynamic take profit level based on the current price
        if side == 'BUY':  # Long position
            # For long positions, we want take profit to trail above the price
            current_take_profit = current_price * (1 + trailing_take_profit_pct)
            current_take_profit = math.floor(current_take_profit * 10**price_precision) / 10**price_precision
            
            # Check if there are open orders specifically for this symbol
            open_orders = self.binance_client.client.futures_get_open_orders(symbol=symbol)
            
            # Find the current take profit order if it exists - only for this specific symbol
            # This is crucial for multi-instance mode to prevent conflicts between different trading pairs
            existing_take_profit = None
            for order in open_orders:
                if (order['symbol'] == symbol and 
                    order['type'] == 'TAKE_PROFIT_MARKET' and 
                    order['side'] == 'SELL'):
                    existing_take_profit = float(order['stopPrice'])
                    break
            
            # If no existing take profit or our new one is better (higher for long), return the new one
            if not existing_take_profit:
                logger.info(f"Long position: Setting initial {self.current_market_condition} take profit to {current_take_profit} ({trailing_take_profit_pct*100}%)")
                logger.info(f"Current price: {current_price}, Entry price: {entry_price}")
                return current_take_profit
            elif current_take_profit > existing_take_profit:
                logger.info(f"Long position: Adjusting {self.current_market_condition} take profit from {existing_take_profit} to {current_take_profit} ({trailing_take_profit_pct*100}%)")
                logger.info(f"Current price: {current_price}, Entry price: {entry_price}, Take profit moved: {existing_take_profit} -> {current_take_profit}")
                return current_take_profit
            else:
                logger.debug(f"Not adjusting trailing take profit for long position: current ({existing_take_profit}) > calculated ({current_take_profit})")
                return None
                
        elif side == 'SELL':  # Short position
            # For short positions, we want take profit to trail below the price
            current_take_profit = current_price * (1 - trailing_take_profit_pct)
            current_take_profit = math.ceil(current_take_profit * 10**price_precision) / 10**price_precision
            
            # Check if there are open orders specifically for this symbol
            open_orders = self.binance_client.client.futures_get_open_orders(symbol=symbol)
            
            # Find the current take profit order if it exists - only for this specific symbol
            # This is crucial for multi-instance mode to prevent conflicts between different trading pairs
            existing_take_profit = None
            for order in open_orders:
                if (order['symbol'] == symbol and
                    order['type'] == 'TAKE_PROFIT_MARKET' and 
                    order['side'] == 'BUY'):
                    existing_take_profit = float(order['stopPrice'])
                    break
            
            # If no existing take profit or our new one is better (lower), return the new one
            if not existing_take_profit:
                logger.info(f"Short position: Setting initial {self.current_market_condition} take profit to {current_take_profit} ({trailing_take_profit_pct*100}%)")
                logger.info(f"Current price: {current_price}, Entry price: {entry_price}")
                return current_take_profit
            elif current_take_profit < existing_take_profit:
                logger.info(f"Short position: Adjusting {self.current_market_condition} take profit from {existing_take_profit} to {current_take_profit} ({trailing_take_profit_pct*100}%)")
                logger.info(f"Current price: {current_price}, Entry price: {entry_price}, Take profit moved: {existing_take_profit} -> {current_take_profit}")
                return current_take_profit
            else:
                logger.debug(f"Not adjusting trailing take profit for short position: current ({existing_take_profit}) < calculated ({current_take_profit})")
                return None
        
        return None
        
    def update_balance_for_compounding(self):
        """Update balance tracking for auto-compounding"""
        if not AUTO_COMPOUND:
            return False
            
        current_balance = self.binance_client.get_account_balance()
        
        # First time initialization
        if self.last_known_balance is None:
            self.last_known_balance = current_balance
            self.initial_balance = current_balance
            return False
        
        profit = current_balance - self.last_known_balance
        
        if profit > 0:
            # We've made profits since last update
            reinvest_amount = profit * COMPOUND_REINVEST_PERCENT
            logger.info(f"Auto-compounding: {reinvest_amount:.2f} USDT from recent {profit:.2f} USDT profit")
            self.last_known_balance = current_balance
            return True
            
        return False

    def calculate_partial_take_profits(self, symbol, side, entry_price):
        """
        Calculate multiple partial take profit levels based on market condition
        
        Args:
            symbol: Trading pair symbol
            side: 'BUY' or 'SELL'
            entry_price: Entry price of the position
            
        Returns:
            list: List of dictionaries with take profit levels and percentages of position to close
        """
        if not USE_TAKE_PROFIT:
            return []
            
        # Choose take profit percentage based on market condition
        # Optimized partial take profits with first target much closer
        # This increases win rate by securing some profit early
        if self.current_market_condition == 'BULLISH':
            tp1_pct = TAKE_PROFIT_PCT_BULLISH * 0.3  # First target at 30% of full target (closer)
            tp2_pct = TAKE_PROFIT_PCT_BULLISH * 0.7  # Second target at 70% of full target
            tp3_pct = TAKE_PROFIT_PCT_BULLISH * 1.2  # Reduced third target from 150% to 120%
        elif self.current_market_condition == 'BEARISH':
            tp1_pct = TAKE_PROFIT_PCT_BEARISH * 0.25  # First target even closer in bearish markets
            tp2_pct = TAKE_PROFIT_PCT_BEARISH * 0.6   # Second target at 60% of full target
            tp3_pct = TAKE_PROFIT_PCT_BEARISH * 1.0   # Last target at 100% (reduced from 130%)
        elif self.current_market_condition == 'SIDEWAYS':
            tp1_pct = TAKE_PROFIT_PCT_SIDEWAYS * 0.4  # Quicker first take profit (40% of target)
            tp2_pct = TAKE_PROFIT_PCT_SIDEWAYS * 0.8  # Second target at 80% of full target
            tp3_pct = TAKE_PROFIT_PCT_SIDEWAYS * 1.1  # More conservative final target
        else:
            tp1_pct = TAKE_PROFIT_PCT * 0.3   # Close to standard settings but more conservative
            tp2_pct = TAKE_PROFIT_PCT * 0.7
            tp3_pct = TAKE_PROFIT_PCT * 1.2
        
        # Get symbol info for price precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        price_precision = 2  # Default
        if symbol_info:
            price_precision = symbol_info.get('price_precision', 2)
        
        # Calculate take profit prices
        if side == "BUY":  # Long position
            tp1_price = round(entry_price * (1 + tp1_pct), price_precision)
            tp2_price = round(entry_price * (1 + tp2_pct), price_precision)
            tp3_price = round(entry_price * (1 + tp3_pct), price_precision)
        else:  # Short position
            tp1_price = round(entry_price * (1 - tp1_pct), price_precision)
            tp2_price = round(entry_price * (1 - tp2_pct), price_precision)
            tp3_price = round(entry_price * (1 - tp3_pct), price_precision)
        
        # Define partial take profit levels with % of position to close at each level
        take_profits = [
            {'price': tp1_price, 'percentage': 0.3, 'pct_from_entry': tp1_pct * 100},  # Close 30% at first TP
            {'price': tp2_price, 'percentage': 0.4, 'pct_from_entry': tp2_pct * 100},  # Close 40% at second TP
            {'price': tp3_price, 'percentage': 0.3, 'pct_from_entry': tp3_pct * 100}   # Close 30% at third TP
        ]
        
        logger.info(f"Calculated {self.current_market_condition} partial take profits: "
                   f"TP1: {tp1_price} ({tp1_pct*100:.2f}%), "
                   f"TP2: {tp2_price} ({tp2_pct*100:.2f}%), "
                   f"TP3: {tp3_price} ({tp3_pct*100:.2f}%)")
                   
        return take_profits
        
    def calculate_volatility_based_stop_loss(self, symbol, side, entry_price, klines=None):
        """
        Enhanced volatility-based stop loss with dynamic multipliers and key level detection
        
        Args:
            symbol: Trading pair symbol
            side: 'BUY' or 'SELL'
            entry_price: Entry price
            klines: Optional recent price data for ATR calculation
            
        Returns:
            float: Volatility-adjusted stop loss price
        """
        if not USE_STOP_LOSS:
            return None
            
        # If no klines provided, use default percentage-based stop loss
        if klines is None or len(klines) < 20:
            return self.calculate_stop_loss(symbol, side, entry_price)
        
        # Special handling for SUI tokens which have higher volatility
        is_sui = 'SUI' in symbol
            
        try:
            # Convert klines to dataframe for ATR calculation
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert string values to numeric
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col])
                
            # Find support/resistance levels (recent swing points)
            highs = []
            lows = []
            
            # Use a minimum of 50 candles for support/resistance detection if available
            # Increased from 50 to 100 for more accurate level detection
            lookback = min(len(df), 100)
            price_data = df[-lookback:].copy()
            
            # Improved swing high/low detection with strength classification
            for i in range(3, len(price_data) - 3):
                # Enhanced Swing high detection - stronger confirmation required
                if (price_data['high'].iloc[i] > price_data['high'].iloc[i-1] and
                    price_data['high'].iloc[i] > price_data['high'].iloc[i-2] and
                    price_data['high'].iloc[i] > price_data['high'].iloc[i-3] and
                    price_data['high'].iloc[i] > price_data['high'].iloc[i+1] and
                    price_data['high'].iloc[i] > price_data['high'].iloc[i+2] and
                    price_data['high'].iloc[i] > price_data['high'].iloc[i+3]):
                    
                    # Add volume confirmation for stronger levels
                    if price_data['volume'].iloc[i] > price_data['volume_ma'].iloc[i] * 1.2:
                        # Higher volume swing points are more significant
                        highs.append((price_data['high'].iloc[i], 1.5))  # (price, strength)
                    else:
                        highs.append((price_data['high'].iloc[i], 1.0))
                
                # Enhanced Swing low detection - stronger confirmation required
                if (price_data['low'].iloc[i] < price_data['low'].iloc[i-1] and
                    price_data['low'].iloc[i] < price_data['low'].iloc[i-2] and
                    price_data['low'].iloc[i] < price_data['low'].iloc[i-3] and
                    price_data['low'].iloc[i] < price_data['low'].iloc[i+1] and
                    price_data['low'].iloc[i] < price_data['low'].iloc[i+2] and
                    price_data['low'].iloc[i] < price_data['low'].iloc[i+3]):
                    
                    # Add volume confirmation for stronger levels
                    if price_data['volume'].iloc[i] > price_data['volume_ma'].iloc[i] * 1.2:
                        # Higher volume swing points are more significant
                        lows.append((price_data['low'].iloc[i], 1.5))  # (price, strength)
                    else:
                        lows.append((price_data['low'].iloc[i], 1.0))
                        
            # Extract just the price for compatibility with existing code
            highs_prices = [h[0] for h in highs]
            lows_prices = [l[0] for l in lows]
            
            # Calculate various ATR periods for better volatility assessment
            atr_short = ta.volatility.average_true_range(
                df['high'], df['low'], df['close'], window=7
            ).iloc[-1]
            
            atr_medium = ta.volatility.average_true_range(
                df['high'], df['low'], df['close'], window=14
            ).iloc[-1]
            
            atr_long = ta.volatility.average_true_range(
                df['high'], df['low'], df['close'], window=21
            ).iloc[-1]
            
            # Use weighted average ATR with more weight to recent volatility
            atr = (atr_short * 0.5 + atr_medium * 0.3 + atr_long * 0.2)
            
            # Calculate ATR as percentage of price
            atr_pct = atr / entry_price
            
            # Calculate price distance from entry to nearest support/resistance
            nearest_support = None
            nearest_resistance = None
            
            # For long positions, find nearest support below entry price
            if side == "BUY":
                valid_supports = [l for l in lows_prices if l < entry_price]
                if valid_supports:
                    # Find the strongest support level that's not too far away
                    valid_supports_with_dist = [(s, entry_price - s) for s in valid_supports]
                    # Sort by distance (ascending) to find closest supports first
                    valid_supports_with_dist.sort(key=lambda x: x[1])
                    
                    # Use the closest significant support
                    if valid_supports_with_dist:
                        nearest_support = valid_supports_with_dist[0][0]
                        
                        # Check if there's a stronger support not too far away (within 2% extra distance)
                        for s, dist in valid_supports_with_dist[1:]:
                            if dist < valid_supports_with_dist[0][1] * 1.5:  # Within 50% extra distance
                                strength = [lvl[1] for lvl in lows if lvl[0] == s][0]
                                if strength > 1.0:  # If it's a stronger level
                                    nearest_support = s
                                    logger.info(f"Using stronger support at {s} instead of closest at {valid_supports_with_dist[0][0]}")
                                    break
            
            # For short positions, find nearest resistance above entry price
            else:
                valid_resistances = [h for h in highs_prices if h > entry_price]
                if valid_resistances:
                    # Find the strongest resistance level that's not too far away
                    valid_resistances_with_dist = [(r, r - entry_price) for r in valid_resistances]
                    # Sort by distance (ascending) to find closest resistances first
                    valid_resistances_with_dist.sort(key=lambda x: x[1])
                    
                    # Use the closest significant resistance
                    if valid_resistances_with_dist:
                        nearest_resistance = valid_resistances_with_dist[0][0]
                        
                        # Check if there's a stronger resistance not too far away (within 2% extra distance)
                        for r, dist in valid_resistances_with_dist[1:]:
                            if dist < valid_resistances_with_dist[0][1] * 1.5:  # Within 50% extra distance
                                strength = [lvl[1] for lvl in highs if lvl[0] == r][0]
                                if strength > 1.0:  # If it's a stronger level
                                    nearest_resistance = r
                                    logger.info(f"Using stronger resistance at {r} instead of closest at {valid_resistances_with_dist[0][0]}")
                                    break
            
            # Significantly increased multipliers to give trades much more room to breathe
            if self.current_market_condition == 'EXTREME_BULLISH':
                atr_multiplier = 4.5  # Increased from 3.0 to 4.5 for much wider stops in extreme bullish trend
            elif self.current_market_condition == 'BULLISH':
                atr_multiplier = 4.0  # Increased from 2.5 to 4.0 for much wider stops in bullish trend
            elif self.current_market_condition == 'EXTREME_BEARISH':
                atr_multiplier = 3.5  # Increased from 2.0 to 3.5 for much wider stops in extreme bearish trend
            elif self.current_market_condition == 'BEARISH':
                atr_multiplier = 3.8  # Increased from 2.2 to 3.8 for much wider stops in bearish trend
            else:  # SIDEWAYS
                atr_multiplier = 3.2  # Increased from 1.8 to 3.2 for much wider stops in sideways market
            
            # Apply SUI-specific adjustments
            if is_sui:
                original_multiplier = atr_multiplier
                atr_multiplier = atr_multiplier * 1.8  # Increased from 1.5 to 1.8 for wider stops on SUI tokens
                logger.info(f"SUI token detected: Increasing ATR multiplier from {original_multiplier} to {atr_multiplier}")
            
            # Calculate stop loss price - use support/resistance levels if available
            if side == "BUY":  # Long
                # Base stop on ATR initially
                atr_stop_distance = atr * atr_multiplier
                atr_stop_price = entry_price - atr_stop_distance
                
                # For long positions, consider nearest support level
                if nearest_support:
                    support_distance = entry_price - nearest_support
                    # Increased the max distance to 2.5x ATR (from 1.5x) to consider stronger support levels
                    if support_distance <= atr_stop_distance * 2.5:
                        # Place stop below support with a much larger buffer (40% of ATR)
                        support_stop_price = nearest_support - (atr * 0.4)
                        
                        # Only use support-based stop if it provides enough distance from entry
                        min_distance_pct = 0.028  # Increased minimum distance from 1.5% to 2.8%
                        min_stop_price = entry_price * (1 - min_distance_pct)
                        
                        if support_stop_price > min_stop_price:
                            # Use the better (higher) of the two stop prices
                            stop_price = max(atr_stop_price, support_stop_price)
                            logger.info(f"Using support-based stop loss: {stop_price} (support level: {nearest_support})")
                        else:
                            logger.info(f"Support too close to entry price, using ATR-based stop: {atr_stop_price}")
                            stop_price = atr_stop_price
                    else:
                        logger.info(f"Support too far from entry ({support_distance:.6f} > {atr_stop_distance * 2.5:.6f}), using ATR-based stop")
                        stop_price = atr_stop_price
                else:
                    logger.info(f"No valid support detected, using ATR-based stop: {atr_stop_price}")
                    stop_price = atr_stop_price
                    
                # Cap maximum stop distance to standard percentage stop loss * multiplier
                # But using much more generous values to prevent premature stop hits
                max_stop_pct = STOP_LOSS_PCT * 1.2  # Base increase of 20%
                if self.current_market_condition == 'EXTREME_BEARISH':
                    max_stop_pct = STOP_LOSS_PCT * 1.1  # Increased from 0.7 to 1.1 for wider stops
                elif self.current_market_condition == 'BEARISH':
                    max_stop_pct = STOP_LOSS_PCT * 1.15  # Increased from 0.8 to 1.15 for wider stops
                
                # For SUI, adapt the maximum stop distance
                if is_sui:
                    max_stop_pct = max_stop_pct * 1.7  # Increased from 1.15 to 1.7 for much wider stops
                    
                max_stop_distance = entry_price * max_stop_pct
                min_stop_price = entry_price - max_stop_distance
                
                # Ensure stop price doesn't exceed max distance
                stop_price = max(stop_price, min_stop_price)
                
            else:  # Short
                # Base stop on ATR initially
                atr_stop_distance = atr * atr_multiplier
                atr_stop_price = entry_price + atr_stop_distance
                
                # For short positions, consider nearest resistance level
                if nearest_resistance:
                    resistance_distance = nearest_resistance - entry_price
                    # Increased the max distance to 2.5x ATR (from 1.5x) to consider stronger resistance levels
                    if resistance_distance <= atr_stop_distance * 2.5:
                        # Place stop above resistance with a much larger buffer (40% of ATR) 
                        resistance_stop_price = nearest_resistance + (atr * 0.4)
                        
                        # Only use resistance-based stop if it provides enough distance from entry
                        min_distance_pct = 0.028  # Increased minimum distance from 1.5% to 2.8%
                        min_stop_price = entry_price * (1 + min_distance_pct)
                        
                        if resistance_stop_price < min_stop_price:
                            # Use the better (lower) of the two stop prices
                            stop_price = min(atr_stop_price, resistance_stop_price)
                            logger.info(f"Using resistance-based stop loss: {stop_price} (resistance level: {nearest_resistance})")
                        else:
                            logger.info(f"Resistance too close to entry price, using ATR-based stop: {atr_stop_price}")
                            stop_price = atr_stop_price
                    else:
                        logger.info(f"Resistance too far from entry ({resistance_distance:.6f} > {atr_stop_distance * 2.5:.6f}), using ATR-based stop")
                        stop_price = atr_stop_price
                else:
                    logger.info(f"No valid resistance detected, using ATR-based stop: {atr_stop_price}")
                    stop_price = atr_stop_price
                
                # Cap maximum stop distance to standard percentage stop loss * multiplier
                # But using much more generous values to prevent premature stop hits
                max_stop_pct = STOP_LOSS_PCT * 1.2  # Base increase of 20%
                if self.current_market_condition == 'EXTREME_BULLISH':
                    max_stop_pct = STOP_LOSS_PCT * 1.1  # Increased from 0.7 to 1.1 for wider stops
                elif self.current_market_condition == 'BULLISH':
                    max_stop_pct = STOP_LOSS_PCT * 1.15  # Increased from 0.8 to 1.15 for wider stops
                
                # For SUI, adapt the maximum stop distance
                if is_sui:
                    max_stop_pct = max_stop_pct * 1.7  # Increased from 1.15 to 1.7 for much wider stops
                    
                max_stop_distance = entry_price * max_stop_pct
                max_stop_price = entry_price + max_stop_distance
                
                # Ensure stop price doesn't exceed max distance
                stop_price = min(stop_price, max_stop_price)
            
            # Apply price precision
            symbol_info = self.binance_client.get_symbol_info(symbol)
            if symbol_info:
                price_precision = symbol_info['price_precision']
                stop_price = round(stop_price, price_precision)
                
            # Add detailed log information for better understanding
            if is_sui:
                logger.info(f"Calculated optimized SUI-specific stop loss at {stop_price} "
                          f"(ATR: {atr:.6f}, {atr_pct*100:.2f}% of price, "
                          f"Multiplier: {atr_multiplier}, Market condition: {self.current_market_condition})")
            else:
                logger.info(f"Calculated optimized ATR-based stop loss at {stop_price} "
                          f"(ATR: {atr:.6f}, {atr_pct*100:.2f}% of price, "
                          f"Multiplier: {atr_multiplier}, Market condition: {self.current_market_condition})")
            return stop_price
                
        except Exception as e:
            logger.error(f"Error calculating volatility-based stop loss: {e}")
            
        # Fall back to standard stop loss if ATR calculation fails
        return self.calculate_stop_loss(symbol, side, entry_price)


def round_step_size(quantity, step_size):
    """Round quantity based on step size"""
    precision = int(round(-math.log10(step_size)))
    return round(math.floor(quantity * 10**precision) / 10**precision, precision)


def get_step_size(min_qty):
    """Get step size from min_qty"""
    step_size = min_qty
    # Handle cases where min_qty is not the step size (common in Binance)
    if float(min_qty) > 0:
        step_size = float(min_qty)
        
    return step_size