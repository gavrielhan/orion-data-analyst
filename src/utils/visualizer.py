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
    
    def create_chart(self, df: pd.DataFrame, chart_type: str, 
                    x_col: Optional[str] = None, y_col: Optional[str] = None,
                    title: Optional[str] = None) -> Optional[str]:
        """
        Create and save a chart based on type and data.
        Returns path to saved file or None if creation fails.
        """
        try:
            plt.figure(figsize=(10, 6))
            
            # Auto-detect columns if not specified
            if x_col is None and len(df.columns) > 0:
                x_col = df.columns[0]
            if y_col is None and len(df.columns) > 1:
                y_col = df.columns[1]
            
            chart_type = chart_type.lower()
            
            if chart_type == "bar":
                self._create_bar_chart(df, x_col, y_col, title)
            elif chart_type == "line":
                self._create_line_chart(df, x_col, y_col, title)
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
                self._create_bar_chart(df, x_col, y_col, title)
            
            # Save and close
            filepath = self._generate_filename(chart_type)
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            plt.close()
            return None
    
    def _create_bar_chart(self, df: pd.DataFrame, x_col: str, y_col: str, title: str):
        """Create bar chart."""
        # Limit to top 15 items for readability
        plot_df = df.head(15) if len(df) > 15 else df
        sns.barplot(data=plot_df, x=x_col, y=y_col, hue=y_col, palette="viridis", legend=False)
        plt.xticks(rotation=45, ha='right')
        plt.title(title or f"{y_col} by {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
    
    def _create_line_chart(self, df: pd.DataFrame, x_col: str, y_col: str, title: str):
        """Create line chart for trends."""
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

