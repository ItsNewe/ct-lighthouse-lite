import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

logo_path: str = "static/favicon.png"

st.set_page_config(page_title="Lighthouse",
    page_icon="static/favicon.png" if os.path.exists(logo_path) else "🛰️",
    layout="wide")  # Use wide layout for dashboard
# Create header with logo and title
header_col1, header_col2 = st.columns([0.1, 0.9])
with header_col1:
    # Check if logo exists before displaying
    if os.path.exists(logo_path):
        st.image(logo_path)
with header_col2:
    st.title("Lighthouse - Trading Performance Dashboard")

#! Unused for now
#db = st.connection('mysql', type="sql")

# Add parameter inputs in the sidebar
st.sidebar.header("Account Parameters")
initial_account_size = st.sidebar.number_input("Initial Account Size (€)", min_value=1000, value=40000, step=1000)
max_daily_loss = st.sidebar.number_input("Max Daily Loss (€)", min_value=100, value=2000, step=100)
max_total_loss = st.sidebar.number_input("Max Total Loss (€)", min_value=1000, value=4000, step=500)

fileUpload = st.file_uploader("Upload a trading data CSV file", type=["csv"])
if fileUpload is not None:
    df = pd.read_csv(fileUpload)
    processed_df = df.copy()

    column_mapping = {
        'Order ID': 'order_id',
        'Position ID': 'position_id',
        'Event': 'event_type',
        'Time (UTC+2)': 'event_time',
        'Volume': 'volume',
        'Quantity': 'qty',
        'Type': 'trade_type',
        'Entry price': 'entry_price',
        'TP': 'take_profit',
        'SL': 'stop_loss',
        'Closing price': 'closing_price',
        'Gross profit': 'gross_profit',
        'Pips': 'pips',
        'Balance': 'balance',
        'Equity': 'equity',
        'Serial #': 'serial_number'
    }
    processed_df.rename(columns=column_mapping, inplace=True)

    processed_df['event_time'] = pd.to_datetime(
        processed_df['event_time'], 
        format='%d/%m/%Y %H:%M:%S.%f'
    )
    
    processed_df['weekday'] = processed_df['event_time'].dt.day_name()
    processed_df['hour'] = processed_df['event_time'].dt.hour
    processed_df['date'] = processed_df['event_time'].dt.date
    
    # Extract numerical values from string columns
    processed_df['qty'] = processed_df['qty'].str.extract(r'([\d.]+)').astype(float)
    processed_df['volume'] = processed_df['volume'].astype(str).str.extract(r'(\d+)').astype(int)
    
    # Convert monetary columns to numeric, handling European formatting
    for col in ['gross_profit', 'balance', 'equity']:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(
                processed_df[col].astype(str).str.replace(r'\s', '', regex=True),
                errors='coerce'
            )
    
    # Convert price columns to numeric
    for col in ['entry_price', 'closing_price', 'take_profit', 'stop_loss']:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    
    # Convert position ID to integer, handling empty strings
    processed_df['position_id'] = pd.to_numeric(processed_df['position_id'], errors='coerce').fillna(0).astype(int)
    processed_df['serial_number'] = pd.to_numeric(processed_df['serial_number'], errors='coerce').astype(int)
    
    # Filter for completed trades (Take Profit/Stop Loss/Position closed events)
    completed_trades = processed_df[processed_df['event_type'].isin(['Take Profit Hit', 'Stop Loss Hit', 'Position closed'])]
    
    # Calculate additional metrics
    
    # 1. Profit Factor = Gross Profits / Gross Losses
    winning_trades = completed_trades[completed_trades['gross_profit'] > 0]
    losing_trades = completed_trades[completed_trades['gross_profit'] < 0]
    
    gross_profit = winning_trades['gross_profit'].sum() if len(winning_trades) > 0 else 0
    gross_loss = abs(losing_trades['gross_profit'].sum()) if len(losing_trades) > 0 else 1  # Avoid div by 0
    
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    # 2. Calculate Sharpe Ratio (daily)
    # Group trades by date to get daily returns
    daily_returns = completed_trades.groupby('date')['gross_profit'].sum()
    
    # Calculate Sharpe ratio (annualized, assuming 252 trading days)
    risk_free_rate = 0  # Simplified, can be adjusted
    if len(daily_returns) > 1:
        sharpe_ratio = np.sqrt(252) * (daily_returns.mean() - risk_free_rate) / daily_returns.std() if daily_returns.std() != 0 else 0
    else:
        sharpe_ratio = 0
    
    # 3. ROI calculation
    # Use actual initial balance or the provided parameter
    initial_balance = processed_df['balance'].iloc[-1] if initial_account_size == 0 else initial_account_size
    current_balance = completed_trades['balance'].iloc[-1] if len(completed_trades) > 0 else initial_balance
    roi = ((current_balance - initial_balance) / initial_balance) * 100
    
    # 4. Maximum Drawdown calculation
    time_data = processed_df.sort_values('event_time')
    balance_equity_df = time_data[['event_time', 'balance', 'equity']].dropna()
    
    # Calculate running maximum and drawdown
    if len(balance_equity_df) > 0:
        balance_equity_df['running_max'] = balance_equity_df['equity'].cummax()
        balance_equity_df['drawdown'] = (balance_equity_df['equity'] - balance_equity_df['running_max']) / balance_equity_df['running_max'] * 100
        max_drawdown = balance_equity_df['drawdown'].min()
        max_drawdown_date = balance_equity_df.loc[balance_equity_df['drawdown'].idxmin(), 'event_time']
    else:
        max_drawdown = 0
        max_drawdown_date = None
    
    # Create tabs for organization
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Time Analysis", "Position Analysis", "Raw Data"])
    
    with tab1:
        st.header("Trading Overview")
        
        # First row: Core metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Total Trades", 
                len(completed_trades),
                help="Total number of completed trades"
            )
        with col2:
            total_pl = completed_trades['gross_profit'].sum()
            # More nuanced color based on profit amount relative to account size
            if total_pl > initial_account_size * 0.05:  # Profit > 5% of account
                profit_icon = "🔥" 
                delta_color = "normal"
            elif total_pl > 0:
                profit_icon = "📈" 
                delta_color = "normal"
            elif total_pl > -initial_account_size * 0.02:  # Small loss < 2%
                profit_icon = "⚠️"
                delta_color = "inverse"
            else:  # Significant loss
                profit_icon = "📉"
                delta_color = "inverse"
                
            st.metric(
                f"{profit_icon} Total Profit/Loss", 
                f"€{total_pl:.2f}",
                delta=f"{(total_pl/initial_account_size)*100:.1f}% of capital" if initial_account_size > 0 else None,
                delta_color='normal',
                help="Sum of all trade profits and losses"
            )
        with col3:
            win_rate = (completed_trades['gross_profit'] > 0).mean() * 100
            
            # Dynamic win rate icons based on thresholds
            if win_rate >= 60:
                win_icon = "🌟"  
            elif win_rate >= 50:
                win_icon = "✅"  
            elif win_rate >= 40:
                win_icon = "⚠️" 
            else:
                win_icon = "❌" 
                
            st.metric(
                f"{win_icon} Win Rate", 
                f"{win_rate:.1f}%",
                delta=f"{win_rate-50:.1f}% vs 50%" if win_rate != 50 else None,
                delta_color='off',
                help="Percentage of trades that were profitable"
            )
        with col4:
            avg_profit = completed_trades['gross_profit'].mean()
            st.metric(
                "Avg. P/L per Trade", 
                f"€{avg_profit:.2f}",
                delta_color="normal" if avg_profit > 0 else "inverse",
                help="Average profit/loss per trade"
            )
            
        # Second row: Advanced metrics  
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            # Dynamic profit factor with more thresholds
            if profit_factor > 2:
                pf_icon = "🏆" 
                pf_delta = f"+{profit_factor-1:.2f} above breakeven"
            elif profit_factor > 1:
                pf_icon = "✅"
                pf_delta = f"+{profit_factor-1:.2f} above breakeven"
            elif profit_factor > 0.8:
                pf_icon = "⚠️"
                pf_delta = f"{profit_factor-1:.2f} from breakeven"
            else:
                pf_icon = "❌"
                pf_delta = f"{profit_factor-1:.2f} from breakeven"
                
            st.metric(
                f"{pf_icon} Profit Factor", 
                f"{profit_factor:.2f}",
                delta=pf_delta,
                delta_color="normal" if profit_factor > 1 else "inverse",
                help="Gross profit divided by gross loss (>1 is profitable)"
            )
        with col2:
            # Dynamic Sharpe ratio with more thresholds
            if sharpe_ratio > 2:
                sharpe_icon = "🌟"  
            elif sharpe_ratio > 1:
                sharpe_icon = "🔥"  
            elif sharpe_ratio > 0:
                sharpe_icon = "⚠️"  
            else:
                sharpe_icon = "❄️"
                
            st.metric(
                f"{sharpe_icon} Sharpe Ratio", 
                f"{sharpe_ratio:.2f}",
                delta_color="normal" if sharpe_ratio > 1 else "inverse",
                help="Risk-adjusted return (higher is better)"
            )
        with col3:
            # ROI doesn't need an icon, the number speaks for itself
            st.metric(
                "Return on Investment", 
                f"{roi:.2f}%",
                delta_color="normal" if roi > 0 else "inverse",
                help="Return on investment percentage"
            )
        with col4:
            # Dynamic drawdown thresholds
            if max_drawdown > -5:
                dd_icon = "🛡️"  # Minimal drawdown
                dd_color = "normal"  
            elif max_drawdown > -15:
                dd_icon = "📉"  # Moderate drawdown
                dd_color = "inverse"  
            else:
                dd_icon = "⚠️"  # Severe drawdown
                dd_color = "inverse"  
                
            st.metric(
                f"{dd_icon} Max Drawdown", 
                f"{max_drawdown:.2f}%",
                delta_color=dd_color,
                help="Maximum percentage drop from peak to trough"
            )
        
        # Third row: Trading statistics - showing fewer icons here
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_win = winning_trades['gross_profit'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['gross_profit'].mean() if len(losing_trades) > 0 else 0
            
            st.metric(
                "Avg Win", 
                f"€{avg_win:.2f}",
                delta=None,
                help="Average profit on winning trades"
            )
        with col2:
            st.metric(
                "Avg Loss", 
                f"€{avg_loss:.2f}",
                delta=None,
                help="Average loss on losing trades"
            )
        with col3:
            win_loss_ratio = abs(avg_win/avg_loss) if avg_loss != 0 else float('inf')
            
            # Only use color for the ratio, no icon needed
            st.metric(
                "Win/Loss Ratio", 
                f"{win_loss_ratio:.2f}" if win_loss_ratio != float('inf') else "∞",
                delta=f"{win_loss_ratio-1:.2f} vs 1.0" if win_loss_ratio != float('inf') and win_loss_ratio != 1 else None,
                delta_color="normal" if win_loss_ratio > 1 else "inverse",
                help="Ratio of average win to average loss (>1 is better)"
            )
        
        # Fourth row: Win/Loss statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            st.metric(
                "Win/Loss Count", 
                f"{win_count}/{loss_count}",
                delta=f"{win_count-loss_count}" if win_count != loss_count else None,
                delta_color="normal" if win_count > loss_count else "inverse",
                help="Number of winning trades vs losing trades"
            )
        with col2:
            expectancy = (win_rate/100 * avg_win) + ((1-win_rate/100) * avg_loss)
            
            # No icon needed here, just color
            st.metric(
                "Expectancy", 
                f"€{expectancy:.2f}",
                delta="positive" if expectancy > 0 else "negative",
                delta_color="normal" if expectancy > 0 else "inverse",
                help="Expected profit/loss per trade"
            )
        with col3:
            # Daily variance
            if len(daily_returns) > 1:
                daily_std = daily_returns.std()
                st.metric(
                    "Daily Volatility", 
                    f"€{daily_std:.2f}",
                    help="Standard deviation of daily returns"
                )
        
        # Fifth row: Extreme values - separated for better styling
        if len(completed_trades) > 0:
            col1, col2 = st.columns(2)
            with col1:
                largest_win = completed_trades['gross_profit'].max()
                
                st.metric(
                    "🏆 Largest Win", 
                    f"€{largest_win:.2f}",
                    delta=f"{largest_win/initial_account_size*100:.2f}% of capital" if initial_account_size > 0 else None,
                    delta_color="normal",
                    help="Largest single winning trade"
                )
            with col2:
                largest_loss = completed_trades['gross_profit'].min()
                
                st.metric(
                    "📉 Largest Loss", 
                    f"€{largest_loss:.2f}",
                    delta=f"{largest_loss/initial_account_size*100:.2f}% of capital" if initial_account_size > 0 else None,
                    delta_color="normal",
                    help="Largest single losing trade"
                )
        
        # Risk metrics section
        st.markdown("---")
        st.subheader("Risk Management")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            risk_pct = (max_total_loss / initial_account_size) * 100
            st.metric(
                "🛑 Max Allowed Loss",
                f"€{max_total_loss:,.2f}",
                delta=f"{risk_pct:.1f}% of capital",
                delta_color="off",
                help="Maximum allowed loss from initial capital"
            )
        with col2:
            daily_risk_pct = (max_daily_loss / initial_account_size) * 100
            st.metric(
                "📅 Max Daily Loss",
                f"€{max_daily_loss:,.2f}",
                delta=f"{daily_risk_pct:.1f}% of capital",
                delta_color="off",
                help="Maximum allowed loss in a single day"
            )
        with col3:
            # Calculate actual max daily loss
            if len(daily_returns) > 0:
                actual_max_daily_loss = daily_returns.min()
                daily_loss_pct = (actual_max_daily_loss / initial_account_size) * 100
                over_limit = actual_max_daily_loss < -max_daily_loss
                
                daily_loss_icon = "🚨" if over_limit else "📊"
                daily_loss_delta = f"{daily_loss_pct:.1f}% of capital"
                if over_limit:
                    daily_loss_delta += " ⚠️ OVER LIMIT"
                
                st.metric(
                    f"{daily_loss_icon} Worst Day",
                    f"€{actual_max_daily_loss:.2f}",
                    delta=daily_loss_delta,
                    delta_color="normal",
                    help="Largest loss experienced in a single day"
                )

        # Balance and Equity over time
        st.subheader("Account Performance")

        time_data = processed_df.sort_values('event_time')
        balance_equity_df = time_data[['event_time', 'balance', 'equity']].dropna()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=balance_equity_df['event_time'],
            y=balance_equity_df['equity'],
            name='Equity',
            line=dict(color='blue')
        ))

        if 'balance' in balance_equity_df.columns:
            fig.add_trace(go.Scatter(
                x=balance_equity_df['event_time'],
                y=balance_equity_df['balance'],
                name='Balance',
                line=dict(color='green')
            ))

        # Add max drawdown point
        if max_drawdown_date is not None:
            fig.add_trace(go.Scatter(
                x=[max_drawdown_date],
                y=[balance_equity_df.loc[balance_equity_df['event_time'] == max_drawdown_date, 'equity'].iloc[0]],
                mode='markers',
                marker=dict(color='red', size=10),
                name=f'Max Drawdown: {max_drawdown:.2f}%'
            ))

        # Add max loss line
        max_loss_line = initial_balance - max_total_loss
        fig.add_shape(
            type="line",
            x0=balance_equity_df['event_time'].min(),
            x1=balance_equity_df['event_time'].max(),
            y0=max_loss_line,
            y1=max_loss_line,
            line=dict(color="red", width=2, dash="dash"),
        )
        
        fig.add_annotation(
            x=balance_equity_df['event_time'].min(),
            y=max_loss_line,
            text="Max Loss Limit",
            showarrow=False,
            yshift=10
        )

        # Set y-axis range to start close to the minimum value
        y_min = min(balance_equity_df['equity'].min() * 0.99, max_loss_line * 0.95)  # Include max loss line
        y_max = balance_equity_df['equity'].max() * 1.01  # 1% above maximum

        fig.update_layout(
            title='Account Balance & Equity',
            xaxis_title='Date',
            yaxis_title='Value (€)',
            yaxis=dict(range=[y_min, y_max]),
            height=500,
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.header("Time-Based Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Weekday performance with Buy/Sell breakdown
            st.subheader("Performance by Weekday")
            weekday_trade_stats = completed_trades.pivot_table(
                values='gross_profit', 
                index='weekday',
                columns='trade_type', 
                aggfunc=['sum', 'mean', 'count']
            ).fillna(0)
            
            # Reorder weekdays
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_trade_stats = weekday_trade_stats.reindex(weekday_order)
            
            st.write(weekday_trade_stats)
            
            weekday_fig = px.bar(
                completed_trades,
                x='weekday',
                y='gross_profit',
                color='trade_type',
                category_orders={"weekday": weekday_order},
                title="Profit/Loss by Weekday",
                labels={"gross_profit": "Profit/Loss (€)", "weekday": "Day of Week", "trade_type": "Position Type"}
            )
            
            weekday_fig.update_layout(height=400)
            st.plotly_chart(weekday_fig, use_container_width=True)
        
        with col2:
            st.subheader("Performance by Hour")
            hourly_trade_stats = completed_trades.pivot_table(
                values='gross_profit', 
                index='hour',
                columns='trade_type', 
                aggfunc=['sum', 'mean', 'count']
            ).fillna(0)
            
            st.write(hourly_trade_stats)
            
            hour_fig = px.bar(
                completed_trades,
                x='hour',
                y='gross_profit',
                color='trade_type',
                title="Profit/Loss by Hour",
                labels={"gross_profit": "Profit/Loss (€)", "hour": "Hour of Day", "trade_type": "Position Type"}
            )
            
            hour_fig.update_layout(height=400)
            st.plotly_chart(hour_fig, use_container_width=True)
            
    with tab3:
        st.header("Position Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart of position types
            st.subheader("Position Types Distribution")
            
            position_counts = completed_trades['trade_type'].value_counts()
            
            pie_fig = px.pie(
                values=position_counts.values,
                names=position_counts.index,
                title="Distribution of Trade Types",
                hole=0.3
            )
            
            pie_fig.update_layout(height=400)
            st.plotly_chart(pie_fig, use_container_width=True)
            
        with col2:
            st.subheader("Performance by Position Type")
            
            position_stats = completed_trades.groupby('trade_type')['gross_profit'].agg(['sum', 'mean', 'count'])
            st.write(position_stats)
            
            position_fig = px.bar(
                position_stats.reset_index(),
                x='trade_type',
                y='sum',
                color='trade_type',
                title="Profit/Loss by Position Type",
                labels={"sum": "Profit/Loss (€)", "trade_type": "Position Type"}
            )
            
            position_fig.update_layout(height=400)
            st.plotly_chart(position_fig, use_container_width=True)
    
    with tab4:
        st.header("Raw Trade Data")
        
        # Add a filter for event type
        event_types = ['All'] + sorted(processed_df['event_type'].unique().tolist())
        selected_event_type = st.selectbox("Filter by Event Type", event_types)
        
        if selected_event_type == 'All':
            filtered_df = processed_df
        else:
            filtered_df = processed_df[processed_df['event_type'] == selected_event_type]
        
        st.dataframe(filtered_df)