"""Visualization utilities for data analysis."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple


class Visualizer:
    """
    Creates and saves various chart types for data analysis.
    Saves all visualizations to configured output directory.
    """
    
    def __init__(self):
        from src.config import config
        
        # Set style for professional-looking charts
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
        
        # Create results directory from config
        self.results_dir = Path(config.output_directory)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_filename(self, chart_type: str) -> Path:
        """Generate timestamped filename for chart."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.results_dir / f"{chart_type}_{timestamp}.png"
    
    def _smart_column_selection(self, df: pd.DataFrame, chart_type: str) -> Tuple[str, str]:
        """
        Intelligently select x and y columns based on data types and chart type.
        Returns (x_col, y_col) tuple.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected DataFrame, got {type(df).__name__}")
        
        if len(df.columns) == 0:
            return None, None
        if len(df.columns) == 1:
            return df.columns[0], df.columns[0]
        
        # Analyze columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Try to detect datetime columns from strings
        for col in categorical_cols[:]:
            if any(kw in col.lower() for kw in ['date', 'time', 'year', 'month', 'day']):
                try:
                    pd.to_datetime(df[col].head(10))
                    datetime_cols.append(col)
                    categorical_cols.remove(col)
                except:
                    pass
        
        # Calculate cardinality for categorical columns
        cat_cardinality = {col: df[col].nunique() for col in categorical_cols}
        
        # Select based on chart type
        if chart_type in ['bar', 'pie']:
            # Categorical x-axis, numeric y-axis
            # For bar charts, prefer ordered columns (years, dates) if available
            if chart_type == 'bar':
                x_col = self._select_ordered_column(categorical_cols, df)
                if not x_col:
                    x_col = self._select_categorical_column(categorical_cols, cat_cardinality)
            else:
                x_col = self._select_categorical_column(categorical_cols, cat_cardinality)
            y_col = self._select_numeric_column(numeric_cols, df)
            
        elif chart_type == 'line':
            # Datetime/ordered x-axis, numeric y-axis
            if datetime_cols:
                x_col = datetime_cols[0]
            elif categorical_cols:
                # Use categorical if it looks ordered (dates, months, etc.)
                x_col = self._select_ordered_column(categorical_cols, df)
            else:
                x_col = numeric_cols[0] if numeric_cols else df.columns[0]
            y_col = self._select_numeric_column(numeric_cols, df, exclude=[x_col])
            
        elif chart_type == 'scatter':
            # Two numeric columns
            if len(numeric_cols) >= 2:
                x_col = numeric_cols[0]
                y_col = numeric_cols[1]
            else:
                x_col = df.columns[0]
                y_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                
        elif chart_type == 'box':
            # Single numeric column
            y_col = self._select_numeric_column(numeric_cols, df)
            x_col = y_col
            
        else:
            # Default: categorical x, numeric y
            x_col = self._select_categorical_column(categorical_cols, cat_cardinality)
            y_col = self._select_numeric_column(numeric_cols, df)
        
        # Fallback if selection failed
        if not x_col:
            x_col = df.columns[0]
        if not y_col:
            y_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        return x_col, y_col
    
    def _select_categorical_column(self, cat_cols: list, cardinality: dict) -> Optional[str]:
        """Select the best categorical column (prefer low cardinality, meaningful names)."""
        if not cat_cols:
            return None
        
        # Prefer columns with reasonable cardinality (2-20 unique values)
        good_cardinality = [col for col in cat_cols if 2 <= cardinality.get(col, 0) <= 20]
        if good_cardinality:
            # Prioritize by name semantics
            for col in good_cardinality:
                col_lower = col.lower()
                if any(kw in col_lower for kw in ['name', 'category', 'type', 'status', 'product', 'customer']):
                    return col
            return good_cardinality[0]
        
        # If no good cardinality, use first categorical
        return cat_cols[0]
    
    def _select_numeric_column(self, num_cols: list, df: pd.DataFrame, exclude: list = None) -> Optional[str]:
        """Select the best numeric column (prefer aggregated values, revenue, sales)."""
        if not num_cols:
            return None
        
        exclude = exclude or []
        available = [col for col in num_cols if col not in exclude]
        if not available:
            return num_cols[0] if num_cols else None
        
        # Prioritize by name semantics (revenue, sales, total, amount, price, count)
        priority_keywords = ['revenue', 'sales', 'total', 'amount', 'price', 'count', 'sum']
        for col in available:
            col_lower = col.lower()
            if any(kw in col_lower for kw in priority_keywords):
                return col
        
        # Otherwise, use the column with the largest range (most interesting to visualize)
        ranges = {col: df[col].max() - df[col].min() for col in available if df[col].dtype in ['int64', 'float64']}
        if ranges:
            return max(ranges, key=ranges.get)
        
        return available[0]
    
    def _select_ordered_column(self, cat_cols: list, df: pd.DataFrame) -> Optional[str]:
        """Select categorical column that represents ordered data (dates, months, etc.)."""
        if not cat_cols:
            return None
        
        # Check for date-like or ordered column names
        ordered_keywords = ['date', 'time', 'year', 'month', 'day', 'quarter', 'week']
        for col in cat_cols:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ordered_keywords):
                return col
        
        # Default to first categorical
        return cat_cols[0]
    
    def create_chart(self, df: pd.DataFrame, chart_type: str, 
                    x_col: Optional[str] = None, y_col: Optional[str] = None,
                    title: Optional[str] = None, hue_col: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Create and save a chart based on type and data.
        Returns (filepath, error_message) tuple. If error, filepath is None and error_message explains why.
        """
        try:
            # Validate df is actually a DataFrame
            if not isinstance(df, pd.DataFrame):
                return None, f"Expected DataFrame, got {type(df).__name__}. Please check the data source."
            
            plt.figure(figsize=(10, 6))
            
            chart_type = chart_type.lower()
            
            # Validate columns exist
            if x_col and x_col not in df.columns:
                return None, f"Column '{x_col}' not found in data. Available columns: {', '.join(df.columns)}"
            if y_col and y_col not in df.columns:
                return None, f"Column '{y_col}' not found in data. Available columns: {', '.join(df.columns)}"
            if hue_col and hue_col not in df.columns:
                return None, f"Column '{hue_col}' not found in data. Available columns: {', '.join(df.columns)}"
            
            # Intelligently auto-detect columns if not specified
            if x_col is None or y_col is None:
                x_col, y_col = self._smart_column_selection(df, chart_type)
            
            # Validate selected columns
            if x_col not in df.columns or y_col not in df.columns:
                return None, f"Auto-selected columns invalid. X: {x_col}, Y: {y_col}. Available: {', '.join(df.columns)}"
            
            if chart_type == "bar":
                self._create_bar_chart(df, x_col, y_col, title, hue_col)
            elif chart_type == "line":
                self._create_line_chart(df, x_col, y_col, title, hue_col)
            elif chart_type == "pie":
                self._create_pie_chart(df, x_col, y_col, title)
            elif chart_type == "scatter":
                self._create_scatter_plot(df, x_col, y_col, title)
            elif chart_type == "box":
                self._create_box_plot(df, y_col, title)
            elif chart_type == "candle":
                self._create_candle_plot(df, title)
            else:
                # Default to bar chart
                self._create_bar_chart(df, x_col, y_col, title, hue_col)
            
            # Save and close
            filepath = self._generate_filename(chart_type)
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath), None
            
        except KeyError as e:
            plt.close()
            return None, f"Column error: {str(e)}. Available columns: {', '.join(df.columns)}"
        except ValueError as e:
            plt.close()
            return None, f"Data value error: {str(e)}. Check data types and values."
        except Exception as e:
            plt.close()
            return None, f"Chart creation failed: {str(e)}"
    
    def _create_bar_chart(self, df: pd.DataFrame, x_col: str, y_col: str, title: str, hue_col: Optional[str] = None):
        """Create bar chart with optional grouping."""
        # Limit to top 15 items for readability
        plot_df = df.head(15) if len(df) > 15 else df
        if hue_col and hue_col in df.columns:
            ax = sns.barplot(data=plot_df, x=x_col, y=y_col, hue=hue_col, palette="viridis")
            # Only add legend if there are labeled artists
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                plt.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Use hue=x_col with legend=False to maintain colorful bars without warning
            sns.barplot(data=plot_df, x=x_col, y=y_col, hue=x_col, palette="viridis", legend=False)
        plt.xticks(rotation=45, ha='right')
        plt.title(title or f"{y_col} by {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
    
    def _create_line_chart(self, df: pd.DataFrame, x_col: str, y_col: str, title: str, hue_col: Optional[str] = None):
        """Create line chart with optional grouping."""
        if hue_col and hue_col in df.columns:
            for group in df[hue_col].unique():
                group_df = df[df[hue_col] == group]
                plt.plot(group_df[x_col], group_df[y_col], marker='o', linewidth=2, markersize=6, label=group)
            plt.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.plot(df[x_col], df[y_col], marker='o', linewidth=2, markersize=6)
        plt.xticks(rotation=45, ha='right')
        plt.title(title or f"{y_col} Trend")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True, alpha=0.3)
    
    def _create_pie_chart(self, df: pd.DataFrame, x_col: str, y_col: str, title: str):
        """Create pie chart."""
        # Limit to top 10 for clarity
        plot_df = df.head(10) if len(df) > 10 else df
        plt.pie(plot_df[y_col], labels=plot_df[x_col], autopct='%1.1f%%', startangle=90)
        plt.title(title or f"Distribution of {y_col}")
        plt.axis('equal')
    
    def _create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str, title: str):
        """Create scatter plot."""
        sns.scatterplot(data=df, x=x_col, y=y_col, alpha=0.6, s=100)
        plt.title(title or f"{y_col} vs {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
    
    def _create_box_plot(self, df: pd.DataFrame, y_col: str, title: str):
        """Create box plot for distribution analysis."""
        sns.boxplot(y=df[y_col], palette="Set2")
        plt.title(title or f"Distribution of {y_col}")
        plt.ylabel(y_col)
    
    def _create_candle_plot(self, df: pd.DataFrame, title: str):
        """Create candlestick-style plot (simplified as OHLC bar chart)."""
        # Requires open, high, low, close columns
        required_cols = ['open', 'high', 'low', 'close']
        
        # Try to find matching columns (case-insensitive)
        col_map = {}
        for req in required_cols:
            for col in df.columns:
                if req in col.lower():
                    col_map[req] = col
                    break
        
        if len(col_map) == 4:
            # Create OHLC bar chart
            for i, row in df.head(30).iterrows():  # Limit to 30 for readability
                plt.plot([i, i], [row[col_map['low']], row[col_map['high']]], 
                        color='black', linewidth=1)
                color = 'green' if row[col_map['close']] >= row[col_map['open']] else 'red'
                plt.plot([i, i], [row[col_map['open']], row[col_map['close']]], 
                        color=color, linewidth=4)
            plt.title(title or "OHLC Chart")
            plt.xlabel("Index")
            plt.ylabel("Value")
        else:
            # Fallback to line chart if columns don't match
            self._create_line_chart(df, df.columns[0], df.columns[1], title)
    
    def save_csv(self, df: pd.DataFrame, filename: str = "results") -> str:
        """Save DataFrame to CSV in results directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.results_dir / f"{filename}_{timestamp}.csv"
        df.to_csv(filepath, index=False)
        return str(filepath)

