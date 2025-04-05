import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

SHORT_COLOR = "#00CC96"
LONG_COLOR = "#EF553B"
logo_path: str = "static/favicon.png"

st.set_page_config(page_title="Lighthouse",
    page_icon="static/favicon.png" if os.path.exists(logo_path) else "🛰️",
    layout="wide"
)

# Create header with logo and title
header_col1, header_col2 = st.columns([0.1, 0.9])
with header_col1:
    # Check if logo exists before displaying
    if os.path.exists(logo_path):
        st.image(logo_path)
with header_col2:
    st.title("Lighthouse - cTrader History Analyzer")


def processCsv(file):
    df = pd.DataFrame()
    match Path(file.name).suffix:
        # Check file extension and read accordingly
        case ".xlsx":
            df = pd.read_excel(file, engine='openpyxl')

        case ".xls":
            df = pd.read_excel(file, engine='xlrd')

        case ".csv":
            try:
                df = pd.read_csv(file, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding="ISO-8859-1")
                except Exception as e:
                    st.error(f"Error reading CSV file: {e}")
                    return pd.DataFrame(), pd.DataFrame()

        case ".ods":
            df = pd.read_excel(file, engine='odf')

        case _:
            st.error("Unsupported file format. Please upload a CSV, Excel, or ODS file.")
    
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

    # Map French event types to English for consistent filtering
    event_mapping = {
        'Créer une Position': 'Create Position',
        'Succès du Stop Loss': 'Stop Loss Hit',
        'Succès du Take Profit': 'Take Profit Hit',
        'Position modifiée (S/L)': 'Position modified (S/L)',
        'Position Fermée': 'Position closed',
        'Position fermée': 'Position closed'
    }
    
    if 'event_type' in processed_df.columns:
        processed_df['event_type'] = processed_df['event_type'].map(
            lambda x: event_mapping.get(x, x)
    )
    processed_df['event_time'] = pd.to_datetime(
        processed_df['event_time'], 
        format='mixed' #'%d/%m/%Y %H:%M:%S.%f'
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
    
    # Calculate position duration
    position_durations = processed_df.groupby('position_id').agg({
        'event_time': ['min', 'max']
    })['event_time']
    position_durations['duration'] = position_durations['max'] - position_durations['min']

    # Merge duration back to the main dataframe
    processed_df = processed_df.merge(
        position_durations['duration'].reset_index(),
        on='position_id',
        how='left'
    )
    # Filter for completed trades (Take Profit/Stop Loss/Position closed events)
    return processed_df, processed_df[processed_df['event_type'].isin(['Take Profit Hit', 'Stop Loss Hit', 'Position closed'])]

#! Unused for now
#db = st.connection('mysql', type="sql")

# Add parameter inputs in the sidebar
st.sidebar.header("Account Parameters")
initial_account_size = st.sidebar.number_input("Initial Account Size (€)", min_value=1000, value=40000, step=1000)
max_daily_loss = st.sidebar.number_input("Max Daily Loss (€)", min_value=100, value=2000, step=100)
max_total_loss = st.sidebar.number_input("Max Total Loss (€)", min_value=1000, value=4000, step=500)

processed_df, completed_trades = pd.DataFrame(), pd.DataFrame()

fileUpload = st.file_uploader("Upload a trading data CSV file", type=["csv", "xlsx", "xls", "ods"])
if fileUpload is not None:
    processed_df, completed_trades = processCsv(fileUpload)


load_sample = st.button("Load Sample Data")
if load_sample:
    sample_path = os.path.join(os.path.dirname(__file__), "static", "sample.csv")
    if os.path.exists(sample_path):
        fileUpload = open(sample_path, "rb")
        processed_df, completed_trades = processCsv(fileUpload)
    else:
        st.error("Sample data file not found")


if not completed_trades.empty:
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Time Analysis", "Position Analysis", "Sequence Analysis", "Raw Data"])

    with tab1:
        st.header("Trading Overview")
        st.subheader("Key Metrics")
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
        st.markdown("---")
        st.subheader("Advanced Metrics")
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
        
        # Third row: Trading statistics
        col1, col2, col3 = st.columns(3, border=True)
        with col1:
            avg_win = winning_trades['gross_profit'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['gross_profit'].mean() if len(losing_trades) > 0 else 0
            
            st.metric(
                "Avg Win", 
                f"€{avg_win:.2f}",
                delta=None,
                help="Average profit on winning trades"
            )

            st.metric(
                "Avg Loss", 
                f"€{avg_loss:.2f}",
                delta=None,
                help="Average loss on losing trades"
            )
        with col2:
                largest_win = completed_trades['gross_profit'].max()
                st.metric(
                    "🏆 Largest Win", 
                    f"€{largest_win:.2f}",
                    delta=f"{largest_win/initial_account_size*100:.2f}% of capital" if initial_account_size > 0 else None,
                    delta_color="normal",
                    help="Largest single winning trade"
                )

                largest_loss = completed_trades['gross_profit'].min()
                st.metric(
                    "📉 Largest Loss", 
                    f"€{largest_loss:.2f}",
                    delta=f"{largest_loss/initial_account_size*100:.2f}% of capital" if initial_account_size > 0 else None,
                    delta_color="normal",
                    help="Largest single losing trade"
                )
        with col3:
            win_loss_ratio = abs(avg_win/avg_loss) if avg_loss != 0 else float('inf')
            st.metric("Winning Trades",
                    f"{len(winning_trades)}",
                delta_color="normal" if len(winning_trades) > 0 else "inverse",
                help="Total number of winning trades"
            )
            st.metric("Losing Trades",
                        f"{len(losing_trades)}",
                    delta_color="normal" if len(losing_trades) > 0 else "inverse",
                    help="Total number of losing trades"
                )
        
        # Fourth row: Win/Loss statistics
        col1, col2, col3 = st.columns(3, border=True)
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
        st.markdown("---")
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
        

        # Weekday performance with Buy/Sell breakdown
        st.subheader("Performance by Weekday")
        col1, col2 = st.columns(2)
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        with col1:
            
            weekday_trade_stats = completed_trades.pivot_table(
                values='gross_profit', 
                index='weekday',
                columns='trade_type', 
                aggfunc=['sum', 'mean', 'count']
            ).fillna(0)
            
            # Reorder weekdays
        
            weekday_trade_stats = weekday_trade_stats.reindex(weekday_order)

            # Original chart with buy/sell breakdown
            weekday_fig = px.bar(
                completed_trades,
                x='weekday',
                y='gross_profit',
                color='trade_type',
                color_discrete_map={'Buy': LONG_COLOR, 'Sell': SHORT_COLOR},
                category_orders={"weekday": weekday_order},
                title="Profit/Loss by Weekday (Buy/Sell)",
                labels={"gross_profit": "Profit/Loss (€)", "weekday": "Day of Week", "trade_type": "Position Type"}
            )
            
            weekday_fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=2, opacity=0.5)
            weekday_fig.update_layout(height=400)
            st.plotly_chart(weekday_fig, use_container_width=True)

        with st.expander("Detailed Statistics by Weekday"):
                st.write(weekday_trade_stats)
                st.write("Total Trades:", weekday_trade_stats['count'].sum())
                st.write("Total Profit/Loss (€):", weekday_trade_stats['sum'].sum())
                st.write("Average Profit/Loss per Trade (€):", (weekday_trade_stats['sum'] / weekday_trade_stats['count']).mean())

        
        with col2:
            # New chart with total P/L per day
            daily_total = completed_trades.groupby('weekday')['gross_profit'].sum().reindex(weekday_order)
            total_fig = px.bar(
                x=daily_total.index,
                y=daily_total.values,
                title="Total Profit/Loss by Weekday",
                labels={"x": "Day of Week", "y": "Total Profit/Loss (€)"},
                color=daily_total.values > 0,
                color_discrete_map={True: SHORT_COLOR, False: LONG_COLOR},
            )
            
            total_fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=2, opacity=0.5)
            total_fig.update_layout(height=400)
            st.plotly_chart(total_fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Performance by Hour")

        hourly_trade_stats = completed_trades.pivot_table(
            values='gross_profit', 
            index='hour',
            columns='trade_type', 
            aggfunc=['sum', 'mean', 'count']
        ).fillna(0)
        
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
        with st.expander("Detailed Statistics by Hour"):
            st.write(hourly_trade_stats)
            st.write("Total Trades:", hourly_trade_stats['count'].sum())
            st.write("Total Profit/Loss (€):", hourly_trade_stats['sum'].sum())
            st.write("Average Profit/Loss per Trade (€):", (hourly_trade_stats['sum'] / hourly_trade_stats['count']).mean())
            
    with tab3:
        st.header("Position Analysis")
        
        # Pie chart of position types
        st.subheader("Position Types Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
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
            position_stats = completed_trades.groupby('trade_type')['gross_profit'].agg(['sum', 'mean', 'count'])
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

        # New section for trade pattern analysis
        st.markdown("---")
        st.subheader("Trade Pattern Analysis")

        # Sort trades by profit/loss
        best_trades = completed_trades.nlargest(10, 'gross_profit')
        worst_trades = completed_trades.nsmallest(10, 'gross_profit')

        col3, col4 = st.columns(2)

        with col3:
            st.write("Top 10 Most Profitable Trades")
            st.dataframe(best_trades[['event_time', 'trade_type', 'volume', 'gross_profit', 'duration', 'weekday', 'hour']])
            
            st.write("Patterns in Most Profitable Trades:")
            best_patterns = pd.DataFrame({
            'Average Duration (min)': best_trades['duration'].dt.total_seconds().mean() / 60,
            'Most Common Hour': best_trades['hour'].mode()[0],
            'Most Common Day': best_trades['weekday'].mode()[0], 
            'Average Volume': best_trades['volume'].mean(),
            'Position Type Split': best_trades['trade_type'].value_counts().to_dict()
            }, index=[0])
            
            st.dataframe(best_patterns)

        with col4:
            st.write("Top 10 Least Profitable Trades")
            st.dataframe(worst_trades[['event_time', 'trade_type', 'volume', 'gross_profit', 'duration', 'weekday', 'hour']])
            
            st.write("Patterns in Least Profitable Trades:")
            worst_patterns = pd.DataFrame({
            'Average Duration (min)': worst_trades['duration'].dt.total_seconds().mean() / 60,
            'Most Common Hour': worst_trades['hour'].mode()[0],
            'Most Common Day': worst_trades['weekday'].mode()[0],
            'Average Volume': worst_trades['volume'].mean(),
            'Position Type Split': worst_trades['trade_type'].value_counts().to_dict()
            }, index=[0])
            
            st.dataframe(worst_patterns)

        # Duration analysis
        st.subheader("Trade Duration vs Profit Analysis")
        
        # Convert duration to minutes for better visualization
        duration_profit_fig = px.scatter(
            completed_trades,
            x=completed_trades['duration'].dt.total_seconds() / 60,
            y='gross_profit',
            color='trade_type',
            title="Trade Duration vs Profit",
            labels={
            "x": "Duration (minutes)",
            "y": "Profit/Loss (€)",
            "trade_type": "Position Type"
            }
        )
        
        duration_profit_fig.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(duration_profit_fig, use_container_width=True)

        # Trade volume analysis
        st.subheader("Volume Analysis")
        
        volume_profit_fig = px.scatter(
            completed_trades,
            x='volume',
            y='gross_profit',
            color='trade_type',
            title="Trade Volume vs Profit",
            labels={
            "volume": "Volume",
            "gross_profit": "Profit/Loss (€)",
            "trade_type": "Position Type"
            }
        )
        
        volume_profit_fig.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(volume_profit_fig, use_container_width=True)

        # Add this to tab3 (Position Analysis) after the volume analysis section

        # Trailing Stop Analysis
        st.markdown("---")
        st.subheader("Trailing Stop Analysis")

        # Identify trailing stop events
        trailing_stop_events = processed_df[processed_df['event_type'] == 'Position modified (S/L)']

        if not trailing_stop_events.empty:
            # Count positions that used trailing stops
            positions_with_ts = trailing_stop_events['position_id'].nunique()
            total_positions = processed_df['position_id'].nunique()
            ts_usage_percent = (positions_with_ts / total_positions) * 100 if total_positions > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "🔄 Positions Using Trailing Stops",
                    f"{positions_with_ts}/{total_positions}",
                    delta=f"{ts_usage_percent:.1f}% of all positions",
                    help="Number of positions that used trailing stops"
                )
                
                # Average number of trailing stop adjustments per position
                ts_adjustments_per_position = trailing_stop_events.groupby('position_id').size().mean()
                st.metric(
                    "⚙️ Avg Adjustments Per Position",
                    f"{ts_adjustments_per_position:.1f}",
                    help="Average number of trailing stop adjustments per position"
                )
            
            with col2:
                # Compare profitability of positions with/without trailing stops
                positions_with_ts_list = trailing_stop_events['position_id'].unique()
                completed_with_ts = completed_trades[completed_trades['position_id'].isin(positions_with_ts_list)]
                completed_without_ts = completed_trades[~completed_trades['position_id'].isin(positions_with_ts_list)]
                
                avg_profit_with_ts = completed_with_ts['gross_profit'].mean() if len(completed_with_ts) > 0 else 0
                avg_profit_without_ts = completed_without_ts['gross_profit'].mean() if len(completed_without_ts) > 0 else 0
                
                profit_diff = avg_profit_with_ts - avg_profit_without_ts
                
                st.metric(
                    "💰 Avg P/L With Trailing Stops",
                    f"€{avg_profit_with_ts:.2f}",
                    delta=f"€{profit_diff:.2f} vs without TS",
                    delta_color="normal" if profit_diff > 0 else "inverse",
                    help="Average profit/loss for positions using trailing stops"
                )
                
                # Win rate with trailing stops
                win_rate_with_ts = (completed_with_ts['gross_profit'] > 0).mean() * 100 if len(completed_with_ts) > 0 else 0
                win_rate_without_ts = (completed_without_ts['gross_profit'] > 0).mean() * 100 if len(completed_without_ts) > 0 else 0
                
                st.metric(
                    "🎯 Win Rate With Trailing Stops",
                    f"{win_rate_with_ts:.1f}%",
                    delta=f"{win_rate_with_ts - win_rate_without_ts:.1f}% vs without TS",
                    delta_color="normal" if win_rate_with_ts > win_rate_without_ts else "inverse",
                    help="Win rate for positions using trailing stops"
                )
            
            with col3:
                # Average duration with trailing stops vs without
                avg_duration_with_ts = completed_with_ts['duration'].mean().total_seconds() / 60 if len(completed_with_ts) > 0 else 0
                avg_duration_without_ts = completed_without_ts['duration'].mean().total_seconds() / 60 if len(completed_without_ts) > 0 else 0
                
                st.metric(
                    "⏱️ Avg Duration With Trailing Stops",
                    f"{avg_duration_with_ts:.1f} min",
                    delta=f"{avg_duration_with_ts - avg_duration_without_ts:.1f} min vs without TS",
                    delta_color="off",
                    help="Average position duration with trailing stops"
                )
                
                # Calculate average stop loss tightening
                if len(trailing_stop_events) >= 2:
                    ts_positions = trailing_stop_events.groupby('position_id')
                    avg_tightening_pips = 0
                    position_count = 0
                    
                    for position_id, events in ts_positions:
                        if len(events) >= 2:
                            position_count += 1
                            # Sort by event time to get first and last stop loss value
                            events_sorted = events.sort_values('event_time')
                            first_sl = events_sorted['stop_loss'].iloc[0]
                            last_sl = events_sorted['stop_loss'].iloc[-1]
                            
                            # Get the trade type to determine if tightening is moving up or down
                            trade_type = processed_df[processed_df['position_id'] == position_id]['trade_type'].iloc[0]
                            
                            # For Buy positions, tightening means moving SL up
                            # For Sell positions, tightening means moving SL down
                            if trade_type == 'Buy':
                                tightening = last_sl - first_sl
                            else:  # Sell
                                tightening = first_sl - last_sl
                            
                            avg_tightening_pips += tightening
                    
                    if position_count > 0:
                        avg_tightening_pips = avg_tightening_pips / position_count
                        st.metric(
                            "📏 Avg Stop Loss Tightening",
                            f"{avg_tightening_pips:.1f} pips",
                            help="Average amount the stop loss was tightened by trailing"
                        )
            
            # Visualization of trailing stop adjustments
            st.subheader("Trailing Stop Adjustment Patterns")
            
            # Get time differences between consecutive trailing stop adjustments
            ts_patterns = trailing_stop_events.sort_values(['position_id', 'event_time'])
            ts_patterns['next_event_time'] = ts_patterns.groupby('position_id')['event_time'].shift(-1)
            ts_patterns['time_between_adjustments'] = (ts_patterns['next_event_time'] - ts_patterns['event_time']).dt.total_seconds() / 60
            ts_patterns = ts_patterns.dropna(subset=['time_between_adjustments'])
            
            if not ts_patterns.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution of time between adjustments
                    time_between_fig = px.histogram(
                        ts_patterns,
                        x='time_between_adjustments',
                        title="Time Between Trailing Stop Adjustments",
                        labels={
                            "time_between_adjustments": "Minutes Between Adjustments",
                            "count": "Frequency"
                        },
                        color_discrete_sequence=[SHORT_COLOR]
                    )
                    
                    time_between_fig.update_layout(height=400)
                    st.plotly_chart(time_between_fig, use_container_width=True)
                
                with col2:
                    # Group trailing stop counts by position
                    ts_counts = ts_patterns.groupby('position_id').size().reset_index(name='adjustment_count')
                    
                    # Count distribution
                    count_fig = px.histogram(
                        ts_counts,
                        x='adjustment_count',
                        title="Number of Trailing Stop Adjustments per Position",
                        labels={
                            "adjustment_count": "Number of Adjustments",
                            "count": "Number of Positions"
                        },
                        color_discrete_sequence=[LONG_COLOR]
                    )
                    
                    count_fig.update_layout(height=400)
                    st.plotly_chart(count_fig, use_container_width=True)
            
            # Detailed trailing stop statistics
            with st.expander("Detailed Trailing Stop Statistics"):
                # Most active positions with trailing stops
                most_active_ts = trailing_stop_events.groupby('position_id').size().sort_values(ascending=False).reset_index(name='adjustment_count')
                most_active_ts = most_active_ts.merge(
                    completed_trades[['position_id', 'gross_profit', 'trade_type']],
                    on='position_id',
                    how='left'
                )
                
                st.write("Positions with Most Trailing Stop Adjustments")
                st.dataframe(most_active_ts)
                
                # Analyze if more adjustments correlate with better results
                if len(most_active_ts) > 1:
                    correlation = most_active_ts['adjustment_count'].corr(most_active_ts['gross_profit'])
                    st.metric(
                        "Correlation: Adjustments vs Profit",
                        f"{correlation:.2f}",
                        delta_color="normal" if correlation > 0 else "inverse",
                        help="Correlation between number of trailing stop adjustments and profit. Positive means more adjustments = more profit."
                    )
        else:
            st.info("No trailing stop adjustments found in the dataset. Try uploading a dataset with 'Position modified (S/L)' events.")

    with tab4:
        st.header("Trade Sequence Analysis")
        
        # Calculate trade streaks
        if len(completed_trades) > 0:
            # Sort trades chronologically
            time_sorted_trades = completed_trades.sort_values('event_time')
            
            # Calculate consecutive win/loss streaks
            time_sorted_trades['is_win'] = time_sorted_trades['gross_profit'] > 0
            
            # Initialize streak counters
            current_streak = 1
            max_win_streak = 0
            max_loss_streak = 0
            current_is_win = None
            
            streak_data = []
            
            # Iterate through trades to identify streaks
            for i in range(len(time_sorted_trades)):
                is_win = time_sorted_trades['is_win'].iloc[i]
                
                if i == 0:
                    # First trade initializes the streak
                    current_is_win = is_win
                elif is_win == current_is_win:
                    # Continuing streak
                    current_streak += 1
                else:
                    # Streak ended, record it
                    streak_data.append({
                        'streak_length': current_streak,
                        'is_win': current_is_win,
                        'end_time': time_sorted_trades['event_time'].iloc[i-1]
                    })
                    
                    # Reset streak
                    current_streak = 1
                    current_is_win = is_win
            
            # Add the last streak
            streak_data.append({
                'streak_length': current_streak,
                'is_win': current_is_win,
                'end_time': time_sorted_trades['event_time'].iloc[-1]
            })
            
            # Convert to DataFrame for analysis
            streaks_df = pd.DataFrame(streak_data)
            
            # Calculate streak statistics
            win_streaks = streaks_df[streaks_df['is_win']]
            loss_streaks = streaks_df[~streaks_df['is_win']]
            
            max_win_streak = win_streaks['streak_length'].max() if len(win_streaks) > 0 else 0
            max_loss_streak = loss_streaks['streak_length'].max() if len(loss_streaks) > 0 else 0
            avg_win_streak = win_streaks['streak_length'].mean() if len(win_streaks) > 0 else 0
            avg_loss_streak = loss_streaks['streak_length'].mean() if len(loss_streaks) > 0 else 0
            
            # Display streak metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "🏆 Max Win Streak",
                    f"{max_win_streak}",
                    help="Maximum consecutive winning trades"
                )
                
                st.metric(
                    "📊 Avg Win Streak",
                    f"{avg_win_streak:.1f}",
                    help="Average length of winning streaks"
                )
            
            with col2:
                st.metric(
                    "⚠️ Max Loss Streak",
                    f"{max_loss_streak}",
                    help="Maximum consecutive losing trades"
                )
                
                st.metric(
                    "📊 Avg Loss Streak",
                    f"{avg_loss_streak:.1f}",
                    help="Average length of losing streaks"
                )
                
            with col3:
                # Calculate streak ratio (win/loss)
                streak_ratio = avg_win_streak / avg_loss_streak if avg_loss_streak > 0 else float('inf')
                
                st.metric(
                    "⚖️ Streak Ratio",
                    f"{streak_ratio:.2f}",
                    delta_color="normal" if streak_ratio > 1 else "inverse",
                    help="Ratio of average win streak to average loss streak"
                )
                
                # Calculate number of streaks
                streak_count = len(streaks_df)
                
                st.metric(
                    "🔄 Total Streaks",
                    f"{streak_count}",
                    help="Total number of winning and losing streaks"
                )
            
            # Visualize streaks
            st.subheader("Trading Streaks Over Time")
            
            # Create a DataFrame with streak type and magnitude for visualization
            vis_data = []
            current_time = time_sorted_trades['event_time'].min()
            
            for _, streak in streaks_df.iterrows():
                vis_data.append({
                    'end_time': streak['end_time'],
                    'streak_length': streak['streak_length'] if streak['is_win'] else -streak['streak_length'],
                    'type': 'Win' if streak['is_win'] else 'Loss'
                })
            
            vis_df = pd.DataFrame(vis_data)
            
            # Plot streak visualization
            streak_fig = px.bar(
                vis_df,
                x='end_time',
                y='streak_length',
                color='type',
                title="Win and Loss Streaks Over Time",
                labels={
                    "end_time": "Date",
                    "streak_length": "Streak Length",
                    "type": "Streak Type"
                },
                color_discrete_map={
                    'Win': SHORT_COLOR,
                    'Loss': LONG_COLOR
                }
            )
            
            streak_fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
            streak_fig.update_layout(height=400)
            st.plotly_chart(streak_fig, use_container_width=True)
            
        # Calculate profit per minute
        st.subheader("Profit per Minute Analysis")
        
        if len(completed_trades) > 0:
            # Calculate duration in minutes and profit per minute
            completed_trades['duration_minutes'] = completed_trades['duration'].dt.total_seconds() / 60
            completed_trades['profit_per_minute'] = completed_trades['gross_profit'] / completed_trades['duration_minutes']
            
            # Remove infinite values (trades with zero duration)
            profit_per_min_df = completed_trades[~np.isinf(completed_trades['profit_per_minute'])]
            profit_per_min_df = profit_per_min_df[~np.isnan(profit_per_min_df['profit_per_minute'])]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Display metrics
                avg_ppm = profit_per_min_df['profit_per_minute'].mean()
                median_ppm = profit_per_min_df['profit_per_minute'].median()
                
                st.metric(
                    "Average Profit/Min",
                    f"€{avg_ppm:.2f}/min",
                    delta_color="normal" if avg_ppm > 0 else "inverse",
                    help="Average profit generated per minute in trade"
                )
                
                st.metric(
                    "Median Profit/Min",
                    f"€{median_ppm:.2f}/min",
                    delta_color="normal" if median_ppm > 0 else "inverse",
                    help="Median profit generated per minute in trade"
                )
            
            with col2:
                # Additional metrics
                max_ppm = profit_per_min_df['profit_per_minute'].max()
                min_ppm = profit_per_min_df['profit_per_minute'].min()
                
                st.metric(
                    "Best Profit/Min",
                    f"€{max_ppm:.2f}/min",
                    help="Highest profit per minute"
                )
                
                st.metric(
                    "Worst Profit/Min",
                    f"€{min_ppm:.2f}/min",
                    help="Lowest profit per minute (biggest loss per minute)"
                )
            
            # Scatter plot of Duration vs. Profit/Minute
            ppm_fig = px.scatter(
                profit_per_min_df,
                x='duration_minutes',
                y='profit_per_minute',
                color='trade_type',
                hover_data=['gross_profit', 'event_time'],
                title="Trade Duration vs. Profit per Minute",
                labels={
                    "duration_minutes": "Duration (minutes)",
                    "profit_per_minute": "Profit per Minute (€/min)",
                    "trade_type": "Position Type",
                    "gross_profit": "Total Profit (€)",
                    "event_time": "Trade Date"
                }
            )
            
            ppm_fig.add_hline(y=0, line_dash="dash", line_color="black")
            ppm_fig.update_layout(height=500)
            st.plotly_chart(ppm_fig, use_container_width=True)
            
        # Day of Month Performance Calendar
        st.subheader("Performance by Day of Month")
        
        if len(completed_trades) > 0:
            # Extract day of month
            completed_trades['day_of_month'] = completed_trades['event_time'].dt.day
            completed_trades['month'] = completed_trades['event_time'].dt.month
            completed_trades['year'] = completed_trades['event_time'].dt.year
            
            # Group by day of month
            day_stats = completed_trades.groupby('day_of_month').agg({
                'gross_profit': ['sum', 'mean', 'count'],
                'event_time': 'count'
            })
            
            # Flatten column MultiIndex
            day_stats.columns = ['_'.join(col).strip() for col in day_stats.columns.values]
            day_stats = day_stats.rename(columns={'event_time_count': 'trade_count'})
            day_stats = day_stats.reset_index()
            
            # Create calendar-style visualization
            # We'll create a grid of days 1-31 with color indicating performance
            
            # Calculate max trades per day for sizing
            max_trades = day_stats['trade_count'].max()
            min_trades = day_stats['trade_count'].min()
            
            # Create calendar figure with 31 days (5 rows x 7 cols with some empty cells)
            rows, cols = 5, 7
            
            # Create a blank calendar grid
            calendar_data = []
            
            # Days 1-31 arranged in a calendar layout
            day = 1
            for row in range(rows):
                for col in range(cols):
                    if day <= 31:  # Valid day of month
                        # Get statistics for this day if available
                        day_data = day_stats[day_stats['day_of_month'] == day]
                        if len(day_data) > 0:
                            profit_sum = day_data['gross_profit_sum'].iloc[0]
                            profit_mean = day_data['gross_profit_mean'].iloc[0]
                            trade_count = day_data['trade_count'].iloc[0]
                        else:
                            profit_sum = 0
                            profit_mean = 0
                            trade_count = 0
                            
                        calendar_data.append({
                            'row': row,
                            'col': col,
                            'day': day,
                            'profit_sum': profit_sum,
                            'profit_mean': profit_mean,
                            'trade_count': trade_count,
                            'color': 'green' if profit_sum > 0 else 'red',
                            'size': 10 + (30 * trade_count / max_trades if max_trades > 0 else 0)
                        })
                        day += 1
            
            calendar_df = pd.DataFrame(calendar_data)
            
            # Create a calendar heatmap
            fig = go.Figure()
            
            # Add days with trades as markers
            for _, day_data in calendar_df.iterrows():
                if day_data['trade_count'] > 0:
                    color = SHORT_COLOR if day_data['profit_sum'] > 0 else LONG_COLOR
                    opacity = 0.7
                    
                    # Special emphasis if highly profitable or unprofitable
                    if abs(day_data['profit_sum']) > gross_profit / 10:  # More than 10% of total profit
                        opacity = 1.0
                    
                    fig.add_trace(go.Scatter(
                        x=[day_data['col']],
                        y=[rows - 1 - day_data['row']],  # Invert y-axis for top-to-bottom
                        mode='markers+text',
                        marker=dict(
                            size=day_data['size'],
                            color=color,
                            opacity=opacity,
                            line=dict(width=1, color='black')
                        ),
                        text=str(int(day_data['day'])),
                        textposition='middle center',
                        name=f"Day {int(day_data['day'])}: €{day_data['profit_sum']:.2f} ({int(day_data['trade_count'])} trades)",
                        hoverinfo='name'
                    ))
                else:
                    # Add day number for days without trades
                    fig.add_trace(go.Scatter(
                        x=[day_data['col']],
                        y=[rows - 1 - day_data['row']],
                        mode='text',
                        text=str(int(day_data['day'])),
                        textposition='middle center',
                        marker=dict(opacity=0),
                        name=f"Day {int(day_data['day'])}: No trades",
                        hoverinfo='name'
                    ))
            
            # Set axis ranges
            fig.update_xaxes(range=[-0.5, cols - 0.5], visible=False)
            fig.update_yaxes(range=[-0.5, rows - 0.5], visible=False)
            
            fig.update_layout(
                title="Monthly Calendar: Profit by Day of Month (Size = Number of Trades)",
                showlegend=False,
                height=400,
                width=800,
                margin=dict(l=20, r=20, t=50, b=20),
                plot_bgcolor='rgba(240, 240, 240, 0.5)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed statistics
            with st.expander("Detailed Day of Month Statistics"):
                st.dataframe(day_stats.sort_values('day_of_month'))
    with tab5:
        st.header("Raw Trade Data")
        
        # Add a filter for event type
        event_types = ['All'] + sorted(processed_df['event_type'].unique().tolist())
        selected_event_type = st.selectbox("Filter by Event Type", event_types)
        
        if selected_event_type == 'All':
            filtered_df = processed_df
        else:
            filtered_df = processed_df[processed_df['event_type'] == selected_event_type]
        
        st.dataframe(filtered_df)