
# DataFrame Analysis Tool

This project provides a structured Python-based framework to facilitate DataFrame analysis. It categorizes features by data type, assesses data quality, and offers detailed visual and textual analyses, including correlations with a target variable. The tool is modular and designed to assist developers in gaining insights into their datasets efficiently.

---

## Features

### 1. **Feature Categorization**
- Classifies DataFrame features into categories based on their data types:
  - Numerical: Continuous and discrete values.
  - Categorical: Nominal and ordinal values.
  - Boolean: Binary values.
  - Datetime: Timestamps and date-related data.
  - Text: Free-form text or strings.
  - Mixed: Columns with mixed or complex data types.

### 2. **Data Quality Assessment**
- Identifies missing values and calculates missing value percentages.
- Highlights duplicate rows.
- Provides summary statistics for numerical and categorical features.

### 3. **Visualization**
- Generates feature-specific graphs:
  - Distributions for numerical features.
  - Correlation plots with the target variable.
  - Text-specific visualizations like word clouds and text length distributions.

### 4. **Target Analysis**
- Analyzes the relationship between features and the target variable.
- Computes correlation coefficients and plots feature-specific insights.

### 5. **Text Analysis**
- Summarizes text statistics, such as word counts and unique word counts.
- Visualizes text data using distributions and word clouds.
- Evaluates the relationship between textual features and the target variable.

---

## Project Structure

```
project_directory/
│
├── categorize.py            # Functions for categorizing features by data type.
├── data_quality.py          # Functions for assessing data quality.
├── visualization.py         # Functions for visualizing features.
├── target_analysis.py       # Functions for analyzing the target variable.
├── text_analysis.py         # Functions for analyzing textual data.
├── tests/                   # Unit tests for all modules.
├── datasets/                # Example datasets for testing.
└── README.md                # Project documentation.
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/dataframe-analysis-tool.git
   cd dataframe-analysis-tool
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Import the relevant module:
   ```python
   from categorize import categorize_all_features
   from data_quality import assess_data_quality
   from visualization import analyze_numerical_feature
   from target_analysis import analyze_target_distribution
   from text_analysis import summarize_text_analysis
   ```

2. Perform feature categorization:
   ```python
   categories = categorize_all_features(df)
   print(categories)
   ```

3. Assess data quality:
   ```python
   assess_data_quality(df)
   ```

4. Generate visualizations for a numerical feature:
   ```python
   analyze_numerical_feature(df, "numerical_column_name")
   ```

5. Summarize text features:
   ```python
   summarize_text_analysis(df['text_column'], df['label_column'])
   ```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

--- 

## Contributing

Contributions are welcome. Please open an issue or submit a pull request for any improvements or bug fixes.
