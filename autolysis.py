import os
import sys
import re
import json
import base64
import subprocess
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from rich.console import Console
from dateutil import parser
import chardet
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
import logging

# Initialize console for rich logging
console = Console()

# Configure logging for tenacity
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variable for AI Proxy token
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise EnvironmentError("AIPROXY_TOKEN is not set. Please set it before running the script.")

# Retry settings
def retry_settings_with_logging():
    return retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        before_sleep=before_sleep_log(logger, logging.INFO)
    )

def read_csv(file_path):
    """Read a CSV file with automatic encoding detection."""
    try:
        console.log("[cyan]Detecting file encoding...[/]")
        with open(file_path, 'rb') as file:
            result = chardet.detect(file.read())
            encoding = result['encoding']
        console.log(f"[green]Detected encoding: {encoding}[/]")
        df = pd.read_csv(file_path, encoding=encoding)
        return df
    except Exception as e:
        console.log(f"[red]Error reading the file: {e}[/]")
        sys.exit(1)

def visualize_data(df, output_folder):
    """Generate advanced visualizations and save them in the specified folder."""
    numeric_data = df.select_dtypes(include='number')

    if not numeric_data.empty:
        console.log("[cyan]Generating correlation heatmap...[/]")
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(output_folder, "correlation_heatmap.png"))
        plt.close()

        console.log("[cyan]Generating boxplot...[/]")
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=numeric_data)
        plt.title("Boxplot of Numeric Data")
        plt.savefig(os.path.join(output_folder, "boxplot.png"))
        plt.close()

        console.log("[cyan]Generating histograms...[/]")
        numeric_data.hist(figsize=(12, 10), bins=20, color='teal')
        plt.savefig(os.path.join(output_folder, "histograms.png"))
        plt.close()

    else:
        console.log("[yellow]No numeric data available for visualizations.[/]")

def main():
    console.log("[cyan]Starting script...")
    if len(sys.argv) != 2:
        console.log("[red]Usage: uv run autolysis.py dataset.csv")
        sys.exit(1)

    file_path = sys.argv[1]
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    output_folder = dataset_name

    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        console.log(f"[green]Output folder created: {output_folder}[/]")

    console.log(f"[yellow]Reading file: {file_path}[/]")
    df = read_csv(file_path)
    console.log("[green]Dataframe loaded.[/]")

    # Perform analysis and visualizations
    df = detect_outliers(df)
    df = perform_clustering(df)
    df = perform_pca(df)

    # Pass the output folder to the visualize_data function
    visualize_data(df, output_folder)

    analysis = {
        "shape": df.shape,
        "columns": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "summary_statistics": df.describe(include="all").to_dict(),
    }

    story = create_story(analysis)
    save_results(analysis, story, output_folder)

    console.log("[green]Analysis completed successfully.")

if __name__ == "__main__":
    main()
