# data_consultant_ai
csv, excel 파일 데이터를 업로드하면 EDA, 진단 , 전처리 방법 , 분석 방법론을 자동으로 추천하는 AI 컨설턴트 
https://demoversion1121.streamlit.app/



## 🌟 Features

- **File Upload Support:**
  - Supports multiple file formats (CSV, Excel, JSON, TXT)
  - Automatic file type detection and processing

- **Basic Information:**
  - Dataset shape and size
  - Preview of first and last few rows
  - Detailed information about data types and memory usage

- **Missing Values Analysis:**
  - Detection of missing values
  - Visual representation of missing value distribution
  - Percentage of missing values per column

- **Categorical Data Analysis:**
  - Frequency tables
  - Bar charts and pie charts
  - Distribution visualization for categorical variables

- **Numerical Data Analysis:**
  - Statistical summaries
  - Distribution plots (histograms with density curves)
  - Box plots for understanding data spread

- **Outlier Analysis:**
  - Detection of outliers using IQR method
  - Visual representation through box plots
  - Outlier statistics and percentages

- **Bivariate Analysis:**
  - Relationship analysis between two variables
  - Different plot types based on data types:
    - Numerical vs Numerical: Scatter plots with correlation
    - Numerical vs Categorical: Box plots
    - Categorical vs Categorical: Grouped bar charts

- **Multivariate Analysis:**
  - Correlation matrix heatmap
  - Detailed correlation values
  - Pair plots for numerical variables
