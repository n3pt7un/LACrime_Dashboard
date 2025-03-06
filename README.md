# Los Angeles Crime Analysis Dashboard

This project provides interactive visualization and analysis of crime data in Los Angeles from 2020-2023. It includes a comprehensive interactive dashboard with multiple analysis views.

## Features

- **Temporal Analysis**: Analyze crime trends over time (yearly, monthly, by day of week, by hour)
- **Spatial Analysis**: Explore geographic distribution of crime with heatmaps and area comparisons
- **Crime Type Analysis**: Investigate different types of crimes and their patterns
- **Demographics Analysis**: Understand victim demographics (age, gender, descent)
- **Predictive Insights**: Time series forecasting and pattern detection
- **Data Explorer**: Browse and filter raw crime data
- **Dark Mode Interface**: Modern, responsive design with dark theme for better visualization
- **Advanced Filtering**: Filter across multiple dimensions including time, location, crime type, and demographics

## Requirements

- Python 3.8+
- Required packages are listed in `requirements.txt`

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate sample data (if needed):
   ```bash
   python generate_sample_data.py
   ```

3. Launch the interactive dashboard:
   ```bash
   # Recommended: Use the launcher script
   python run_dashboard.py
   
   # Or run directly (not recommended)
   python dashboard.py
   ```

4. Open your browser and go to http://127.0.0.1:8050/

5. Or explore the Jupyter notebook:
   ```bash
   jupyter notebook sample.ipynb
   ```

## Data Source

Crime data is sourced from the Los Angeles Police Department, via Kaggle dataset "Crimes in Los Angeles 2020-2023" by asaniczka. For testing purposes, the dashboard can also generate realistic sample data that matches the expected schema.

## Dashboard Components

### Temporal Analysis
- Yearly crime trends by category
- Monthly crime patterns
- Crime by hour and day of week
- Seasonal patterns

### Spatial Analysis
- Crime heatmap of Los Angeles
- Top areas by crime count
- Crime categories by area

### Crime Type Analysis
- Crime categories and types treemap
- Top 10 crime types
- Crime evolution over time
- Weapon types used

### Demographics Analysis
- Victim age distribution
- Crime types by victim gender
- Crime categories by victim age group
- Crime categories by victim descent

### Insights & Predictions
- Time series forecasting
- Correlation analysis
- Crime clustering
- Feature importance for predicting violent crimes

### Data Explorer
- Filterable data table
- Export functionality

## Testing

Run the test script to ensure all components are working correctly:
```bash
python dashboard_test_minimal.py
```

## Troubleshooting

If the dashboard doesn't load properly:

1. Check that you have all required dependencies installed
2. Make sure you have data in the `/data` directory or let the dashboard generate sample data
3. Run the minimal test script to verify the basic functionality
4. Check the console output for error messages

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.