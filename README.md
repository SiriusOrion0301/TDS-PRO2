# Automated Data Analysis with Autolysis

## Project Overview
Autolysis is a Python-based automated data analysis tool that leverages the capabilities of GPT-4o-Mini. It processes CSV datasets, performs statistical and exploratory analyses, generates visualizations, and crafts data-driven narratives. The script dynamically analyzes any dataset, minimizes token usage, and outputs insightful Markdown reports with supporting PNG visualizations, ensuring a comprehensive data storytelling experience.

## Key Features

- **Automated Analysis**: 
  - Conducts summary statistics, outlier detection, and correlation analysis.
  - Automatically handles missing values and provides insights into data quality.

- **Visual Insights**: 
  - Generates clear, informative charts, including correlation heatmaps, outlier bar charts, and distribution plots.
  - Visualizations are saved as PNG files for easy sharing and reporting.

- **Data Narratives**: 
  - Crafts meaningful stories based on findings, providing context and insights derived from the analysis.
  - The narrative is generated using advanced AI, ensuring a human-like storytelling approach.

- **Extensibility**: 
  - Works seamlessly with any CSV file, making it adaptable to various datasets.
  - Users can easily modify the script to include additional analyses or visualizations as needed.

- **User-Friendly**: 
  - Simple command-line interface for running analyses.
  - Outputs a comprehensive Markdown report (`README.md`) summarizing the analysis, visualizations, and narratives.

## Getting Started

### Prerequisites
- Python 3.9 or higher
- Required Python packages (install via pip):
  ```bash
  pip install pandas seaborn matplotlib numpy scipy openai scikit-learn requests python-dotenv
  ```

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/autolysis.git
   cd autolysis
   ```

2. **Create a `.env` File**: In the same directory, create a `.env` file and add your AIPROXY token:
   ```
   AIPROXY_TOKEN=your_actual_token_here
   ```

### Usage
To run the analysis, use the following command:
    uv run autolysis.py <path_to_your_dataset.csv>

### Output
- The script will generate:
  - A `README.md` file containing the analysis summary, visualizations, and narratives.
  - PNG files for each visualization in the current directory.

## Conclusion
Autolysis makes data analysis intuitive, scalable, and efficient through intelligent automation and AI-powered insights. Whether you're a data scientist, analyst, or just someone looking to gain insights from data, Autolysis provides a powerful tool to streamline your analysis process.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Special thanks to the developers of the libraries used in this project, including Pandas, Seaborn, Matplotlib, and OpenAI.


