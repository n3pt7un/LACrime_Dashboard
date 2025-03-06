import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, callback, State, dash_table
import dash_bootstrap_components as dbc
import kagglehub
from datetime import datetime
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
import calendar

# Initialize the app with dark theme
app = Dash(
    __name__, 
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

# Set server timeout to longer value
server = app.server

# Cache configuration (important to prevent reloading data)
from flask_caching import Cache

# Setup cache with simple config
cache = Cache(app.server, config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 300
})

# Define age group mapping function
def map_age_group(age):
    if pd.isna(age):
        return "Unknown"
    age = int(age)
    if age < 18:
        return "Under 18"
    elif age < 25:
        return "18-24"
    elif age < 35:
        return "25-34"
    elif age < 45:
        return "35-44"
    elif age < 55:
        return "45-54"
    elif age < 65:
        return "55-64"
    else:
        return "65+"

# Define time of day mapping function
def map_time_of_day(hour):
    if pd.isna(hour):
        return "Unknown"
    hour = int(hour)
    if 5 <= hour < 12:
        return "Morning (5am-12pm)"
    elif 12 <= hour < 17:
        return "Afternoon (12pm-5pm)"
    elif 17 <= hour < 21:
        return "Evening (5pm-9pm)"
    else:
        return "Night (9pm-5am)"

# Define custom dropdown styles to match dark theme
dropdown_style = {
    'backgroundColor': '#303030',
    'color': 'white',
    'border': '1px solid #555',
    'borderRadius': '4px'
}

# Style for dropdown options (menu items)
dropdown_options_style = {
    'backgroundColor': '#303030',
    'color': 'white'
}

# Define season mapping function
def map_season(month):
    if pd.isna(month):
        return "Unknown"
    month = int(month)
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

# Define crime category mapping
def map_crime_category(crime_description):
    violent_crimes = ['ASSAULT', 'ROBBERY', 'HOMICIDE', 'KIDNAPPING', 'RAPE', 'SEX']
    property_crimes = ['BURGLARY', 'THEFT', 'STOLEN', 'VEHICLE', 'VANDALISM', 'ARSON']
    drug_crimes = ['NARCOTICS', 'DRUGS', 'ALCOHOL']
    
    if any(crime in crime_description.upper() for crime in violent_crimes):
        return "Violent Crime"
    elif any(crime in crime_description.upper() for crime in property_crimes):
        return "Property Crime"
    elif any(crime in crime_description.upper() for crime in drug_crimes):
        return "Drug/Alcohol Crime"
    else:
        return "Other Crime"

# Load and process data
def load_data():
    try:
        # Debug output for troubleshooting
        print("Starting to load data...")
        
        # First, try to load from the data directory
        try:
            import os
            
            # Check several possible data file locations
            possible_paths = [
                'data/Crime_Data_from_2020_to_Present2025.csv',
                'data/sample_crime_data.csv',
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'Crime_Data_from_2020_to_Present2025.csv'),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'sample_crime_data.csv')
            ]
            
            data_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    data_path = path
                    print(f"Found data file at {data_path}")
                    break
            
            if data_path:
                df = pd.read_csv(data_path)
                print(f"Successfully loaded data with {len(df)} rows")
            else:
                # If no data file found, create sample dataset
                print("No data file found, generating sample dataset...")
                raise FileNotFoundError("No data file found")
                
        except Exception as load_err:
            print(f"Error loading data file: {load_err}")
            # Create a sample dataset for testing
            print("Creating a sample dataset for testing...")
            
            # Generate sample data
            # Create several area names and crime types for more realistic visualization
            area_names = ['CENTRAL', 'RAMPART', 'SOUTHWEST', 'HOLLENBECK', 'HARBOR', 
                         'HOLLYWOOD', 'WILSHIRE', 'WEST LA', 'PACIFIC', 'NORTHEAST']
            
            crime_descriptions = [
                'BATTERY - SIMPLE ASSAULT', 'THEFT FROM MOTOR VEHICLE', 'BURGLARY',
                'ROBBERY', 'ASSAULT WITH DEADLY WEAPON', 'VANDALISM', 'VEHICLE STOLEN',
                'CRIMINAL THREATS', 'TRESPASSING', 'SHOPLIFTING'
            ]
            
            # Create varied data for better visualization
            df = pd.DataFrame({
                'division_number': range(5000),
                'date_reported': ['2020-01-01'] * 5000,
                'date_occurred': pd.date_range(start='2020-01-01', periods=5000),
                'area': list(range(1, 11)) * 500,
                'area_name': [area_names[i % len(area_names)] for i in range(5000)],
                'reporting_district': [100 + i % 900 for i in range(5000)],
                'part': [1] * 5000,
                'crime_code': [620 + i % 10 for i in range(5000)],
                'crime_description': [crime_descriptions[i % len(crime_descriptions)] for i in range(5000)],
                'modus_operandi': [''] * 5000,
                'victim_age': [15 + i % 70 for i in range(5000)],
                'victim_sex': ['M', 'F', 'X'] * 1667,
                'victim_descent': ['W', 'B', 'H', 'A', 'O'] * 1000,
                'premise_code': [101.0] * 5000,
                'premise_description': ['STREET', 'RESIDENCE', 'PARKING LOT', 'SIDEWALK', 'STORE'] * 1000,
                'weapon_code': [400.0] * 5000,
                'weapon_description': ['STRONG-ARM', 'HANDGUN', 'KNIFE', 'UNKNOWN', 'OTHER'] * 1000,
                'status': ['IC'] * 5000,
                'status_description': ['INVESTIGATION CONTINUED'] * 5000,
                'crime_code_1': [624.0] * 5000,
                'crime_code_2': [np.nan] * 5000,
                'crime_code_3': [np.nan] * 5000,
                'crime_code_4': [np.nan] * 5000,
                'location': ['1ST & SPRING ST'] * 5000,
                'cross_street': [''] * 5000,
                'latitude': [34.0522 + (np.random.random() - 0.5) * 0.1 for _ in range(5000)],
                'longitude': [-118.2437 + (np.random.random() - 0.5) * 0.1 for _ in range(5000)]
            })
            
            # Convert date_occurred to datetime
            df['date_occurred'] = pd.to_datetime(df['date_occurred'])
            
            # Add derived time columns exactly as in the schema
            df['year'] = df['date_occurred'].dt.year
            df['month'] = df['date_occurred'].dt.month
            df['day'] = df['date_occurred'].dt.day
            df['hour'] = df['date_occurred'].dt.hour
            df['day_of_week'] = df['date_occurred'].dt.day_name()
            df['month_name'] = df['date_occurred'].dt.month_name()
            df['year_month'] = df['date_occurred'].dt.strftime('%Y-%m')
            
            # Convert int columns to int32 as specified in the schema
            int32_cols = ['year', 'month', 'day', 'hour']
            for col in int32_cols:
                df[col] = df[col].astype('int32')
                
            print(f"Created sample dataset with {len(df)} rows")
        
        print(f"Columns in the dataset: {df.columns.tolist()}")
        print(f"Data types: {df.dtypes}")
        
        # Make a copy of the dataframe
        df_copy = df.copy()
        # Ensure date_occurred is datetime
        if df_copy['date_occurred'].dtype != 'datetime64[ns]':
            print("Converting date_occurred to datetime...")
            df_copy['date_occurred'] = pd.to_datetime(df_copy['date_occurred'], errors='coerce')
            # Create additional time-based columns
            df_copy["year"] = df_copy["date_occurred"].dt.year
            df_copy["month"] = df_copy["date_occurred"].dt.month
            df_copy["day"] = df_copy["date_occurred"].dt.day
            df_copy["hour"] = df_copy["date_occurred"].dt.hour
            df_copy["day_of_week"] = df_copy["date_occurred"].dt.day_name()
            df_copy["month_name"] = df_copy["date_occurred"].dt.month_name()
            df_copy["year_month"] = df_copy["date_occurred"].dt.strftime('%Y-%m')
            print(f"After conversion: {df_copy['date_occurred'].dtype}")
        
        # Filter out records with invalid lat/lon
        original_count = len(df_copy)
        df_copy = df_copy[(df_copy["latitude"] >= 33.7) & (df_copy["latitude"] <= 34.8) & 
                          (df_copy["longitude"] >= -118.7) & (df_copy["longitude"] <= -117.0)]
        print(f"Filtered latitude/longitude: {original_count} → {len(df_copy)} records")
        
        # We'll add quarter and week_of_year as derived columns
        df_copy["quarter"] = df_copy["date_occurred"].dt.quarter
        df_copy["week_of_year"] = df_copy["date_occurred"].dt.isocalendar().week

        # Derived features
        df_copy["time_of_day"] = df_copy["hour"].apply(map_time_of_day)
        df_copy["season"] = df_copy["month"].apply(map_season)
        df_copy["age_group"] = df_copy["victim_age"].apply(map_age_group)
        df_copy["crime_category"] = df_copy["crime_description"].apply(map_crime_category)

        # Categorical columns (object type)
        categorical_columns = [
            'modus_operandi',
            'victim_sex',
            'victim_descent',
            'premise_description',
            'weapon_description',
            'cross_street'
        ]

        # Numerical columns with missing values
        numerical_columns = [
            'premise_code',
            'weapon_code',
            'crime_code_1',
            'crime_code_2',
            'crime_code_3',
            'crime_code_4'
        ]

        # Handle categorical columns
        for col in categorical_columns:
            # Fill with most frequent value
            df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)

        # Handle numerical columns
        for col in numerical_columns:
            # Strategy depends on the column
            if col in ['crime_code_2', 'crime_code_3', 'crime_code_4']:
                # For these columns with very few non-null values, fill with 0 or a sentinel value
                df_copy[col].fillna(0, inplace=True)
            else:
                # For other numerical columns, use median
                df_copy[col].fillna(df_copy[col].median(), inplace=True)

        # Create victim demographics column for better categorization
        victim_desc_mapping = {
            'A': 'Asian',
            'B': 'Black',
            'C': 'Chinese',
            'D': 'Cambodian',
            'F': 'Filipino',
            'G': 'Guamanian',
            'H': 'Hispanic/Latino',
            'I': 'American Indian/Alaskan Native',
            'J': 'Japanese',
            'K': 'Korean',
            'L': 'Laotian',
            'O': 'Other',
            'P': 'Pacific Islander',
            'S': 'Samoan',
            'U': 'Hawaiian',
            'V': 'Vietnamese',
            'W': 'White',
            'X': 'Unknown',
            'Z': 'Asian Indian'
        }
        df_copy['victim_descent_desc'] = df_copy['victim_descent'].map(victim_desc_mapping).fillna('Unknown')
        
        # Convert victim sex to descriptive values
        victim_sex_mapping = {
            'M': 'Male',
            'F': 'Female',
            'X': 'Unknown'
        }
        df_copy['victim_sex_desc'] = df_copy['victim_sex'].map(victim_sex_mapping).fillna('Unknown')

        # Calculate crime density by area for heatmap
        area_counts = df_copy.groupby('area_name').size().reset_index(name='crime_count')
        df_copy = pd.merge(df_copy, area_counts, on='area_name', how='left')
        
        # Add time_occurred if it doesn't exist (for hover data)
        if 'time_occurred' not in df_copy.columns:
            df_copy['time_occurred'] = df_copy['hour'].apply(lambda x: f"{x:02d}:00")

        return df_copy

    except Exception as e:
        print(f"Error loading data: {e}")
        # Return a small sample dataframe for testing if the data can't be loaded
        return pd.DataFrame()

# Load the data once at startup, not in a callback
print("Calling load_data() function...")
df = load_data()
print(f"Data loaded: DataFrame has {len(df)} rows and {len(df.columns)} columns")
if not df.empty:
    print(f"First few rows: {df.head(2)}")
else:
    print("WARNING: Loaded DataFrame is empty!")

# Record load time to verify we don't reload
import time
DATA_LOAD_TIME = time.strftime("%H:%M:%S")
print(f"Data loaded at: {DATA_LOAD_TIME}")

# Define available filters
crime_types = sorted(df["crime_description"].unique())
crime_categories = sorted(df["crime_category"].unique())
areas = sorted(df["area_name"].unique())
years = sorted(df["year"].unique())
victim_sex = sorted(df['victim_sex_desc'].unique())
victim_desc = sorted(df['victim_descent_desc'].unique()) 
seasons = sorted(df['season'].unique())
times_of_day = sorted(df['time_of_day'].unique())
age_groups = sorted(df['age_group'].unique())

# Create a navbar with title
navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.I(className="fas fa-chart-bar mr-2"), width="auto"),
                        dbc.Col(dbc.NavbarBrand("Los Angeles Crime Dashboard (2020-2025)", className="ml-2")),
                    ],
                    align="center",
                ),
                href="#",
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
        ]
    ),
    color="dark",
    dark=True,
)

# Define custom dropdown styles to match dark theme
dropdown_style = {
    'backgroundColor': '#303030',
    'color': 'white',
    'border': '1px solid #555',
    'borderRadius': '4px'
}

# Style for dropdown options (menu items)
dropdown_options_style = {
    'backgroundColor': '#303030',
    'color': 'white'
}

# Filter panel with collapsible sections
filter_panel = dbc.Card(
    [
        dbc.CardHeader(html.H5("Filters", className="text-center")),
        dbc.CardBody(
            [
                # Time filters
                html.Div(
                    [
                        html.H6("Time Filters", className="mt-2"),
                        html.Hr(className="my-2"),
                        html.P("Select Year:"),
                        dcc.Dropdown(
                            id="year-dropdown",
                            options=[{"label": str(year), "value": year} for year in years],
                            value=None,
                            multi=True,
                            placeholder='All Years',
                            style=dropdown_style,
                            className="dark-dropdown"
                        ),
                        html.P("Select Season:", className="mt-3"),
                        dcc.Dropdown(
                            id="season-dropdown",
                            options=[{"label": season, "value": season} for season in seasons],
                            value=None,
                            multi=True,
                            placeholder="All Seasons",
                            style=dropdown_style,
                            className="dark-dropdown"
                        ),
                        html.P("Select Time of Day:", className="mt-3"),
                        dcc.Dropdown(
                            id="time-of-day-dropdown",
                            options=[{"label": tod, "value": tod} for tod in times_of_day],
                            value=None,
                            multi=True,
                            placeholder="All Times",
                            style=dropdown_style,
                            className="dark-dropdown"
                        ),
                    ]
                ),
                
                # Location filters
                html.Div(
                    [
                        html.H6("Location Filters", className="mt-4"),
                        html.Hr(className="my-2"),
                        html.P("Select Area:"),
                        dcc.Dropdown(
                            id="area-dropdown",
                            options=[{"label": area, "value": area} for area in areas],
                            value=None,
                            multi=True,
                            placeholder="All Areas",
                            style=dropdown_style,
                            className="dark-dropdown"
                        ),
                    ]
                ),
                
                # Crime type filters
                html.Div(
                    [
                        html.H6("Crime Filters", className="mt-4"),
                        html.Hr(className="my-2"),
                        html.P("Select Crime Category:"),
                        dcc.Dropdown(
                            id="crime-category-dropdown",
                            options=[{"label": cat, "value": cat} for cat in crime_categories],
                            value=None,
                            multi=True,
                            placeholder="All Categories",
                            style=dropdown_style,
                            className="dark-dropdown"
                        ),
                        html.P("Select Crime Type:", className="mt-3"),
                        dcc.Dropdown(
                            id="crime-dropdown",
                            options=[{"label": crime, "value": crime} for crime in crime_types],
                            value=None,
                            multi=True,
                            placeholder="All Crime Types",
                            style=dropdown_style,
                            className="dark-dropdown"
                        ),
                    ]
                ),
                
                # Demographic filters
                html.Div(
                    [
                        html.H6("Demographic Filters", className="mt-4"),
                        html.Hr(className="my-2"),
                        html.P("Select Victim Sex:"),
                        dcc.Dropdown(
                            id="victim-sex-dropdown",
                            options=[{"label": sex, "value": sex} for sex in victim_sex],
                            value=None,
                            multi=True,
                            placeholder="All",
                            style=dropdown_style,
                            className="dark-dropdown"
                        ),
                        html.P("Select Victim Descent:", className="mt-3"),
                        dcc.Dropdown(
                            id="victim-descent-dropdown",
                            options=[{"label": desc, "value": desc} for desc in victim_desc],
                            value=None,
                            multi=True,
                            placeholder="All",
                            style=dropdown_style,
                            className="dark-dropdown"
                        ),
                        html.P("Select Age Group:", className="mt-3"),
                        dcc.Dropdown(
                            id="age-group-dropdown",
                            options=[{"label": age, "value": age} for age in age_groups],
                            value=None,
                            multi=True,
                            placeholder="All Age Groups",
                            style=dropdown_style,
                            className="dark-dropdown"
                        ),
                    ]
                ),
                
                # Reset button
                html.Div(
                    dbc.Button("Reset All Filters", id="reset-button", color="secondary", className="w-100 mt-4"),
                    className="d-grid gap-2",
                )
            ]
        ),
    ],
    className="sticky-top",
    style={"height": "calc(100vh - 60px)", "overflowY": "auto"}
)

# Create the app layout
app.layout = html.Div([
    # Store for sharing filter state across callbacks
    dcc.Store(id='filter-store'),
    
    # Navbar
    navbar,
    
    # Main content
    dbc.Container(
        [
            dbc.Row(
                [
                    # Filter panel (collapsible on small screens)
                    dbc.Col(filter_panel, width=12, lg=3, className="mb-4"),
                    
                    # Dashboard content
                    dbc.Col(
                        [
                            # Top KPI row
                            dbc.Row(
                                [
                                    dbc.Col(dbc.Card(id="total-crimes-card", className="text-center h-100"), width=12, md=3),
                                    dbc.Col(dbc.Card(id="most-common-crime-card", className="text-center h-100"), width=12, md=3),
                                    dbc.Col(dbc.Card(id="most-dangerous-area-card", className="text-center h-100"), width=12, md=3),
                                    dbc.Col(dbc.Card(id="time-pattern-card", className="text-center h-100"), width=12, md=3),
                                ],
                                className="mb-4",
                            ),
                            
                            # Main tabs
                            dbc.Tabs(
                                [
                                    # Temporal Analysis
                                    dbc.Tab(
                                        [
                                            html.H4("Crime Trends Over Time", className="mt-3 text-center"),
                                            html.P(
                                                "Analyze how crime patterns change across different time periods.",
                                                className="text-center text-muted mb-4"
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(dcc.Graph(id="yearly-trend-line"), width=12, lg=6),
                                                    dbc.Col(dcc.Graph(id="monthly-trend-line"), width=12, lg=6),
                                                ],
                                                className="mb-4",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(dcc.Graph(id="hour-heatmap"), width=12, lg=6),
                                                    dbc.Col(dcc.Graph(id="day-of-week-chart"), width=12, lg=6),
                                                ],
                                                className="mb-4",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(dcc.Graph(id="seasonal-pattern"), width=12),
                                                ],
                                                className="mb-4",
                                            ),
                                        ],
                                        label="Temporal Analysis",
                                        tab_id="tab-temporal",
                                    ),
                                    
                                    # Spatial Analysis
                                    dbc.Tab(
                                        [
                                            html.H4("Geographic Distribution of Crime", className="mt-3 text-center"),
                                            html.P(
                                                "Explore how crime is distributed across different areas of Los Angeles.",
                                                className="text-center text-muted mb-4"
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.P("Sample Size (for map rendering):"),
                                                            dcc.Slider(
                                                                id="sample-slider",
                                                                min=1000,
                                                                max=50000,
                                                                step=1000,
                                                                value=5000,
                                                                marks={1000: "1K", 10000: "10K", 25000: "25K", 50000: "50K"},
                                                                className="dark-slider"
                                                            ),
                                                        ],
                                                        width=12,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(dcc.Graph(id="crime-heatmap", style={"height": "70vh"}), width=12),
                                                ],
                                                className="mb-4",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(dcc.Graph(id="area-comparison-chart"), width=12, lg=6),
                                                    dbc.Col(dcc.Graph(id="crime-by-area-pie"), width=12, lg=6),
                                                ],
                                                className="mb-4",
                                            ),
                                        ],
                                        label="Spatial Analysis",
                                        tab_id="tab-spatial",
                                    ),
                                    
                                    # Crime Type Analysis
                                    dbc.Tab(
                                        [
                                            html.H4("Crime Categories and Patterns", className="mt-3 text-center"),
                                            html.P(
                                                "Analyze different types of crimes and their characteristics.",
                                                className="text-center text-muted mb-4"
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(dcc.Graph(id="crime-treemap"), width=12, lg=6),
                                                    dbc.Col(dcc.Graph(id="top-crimes-chart"), width=12, lg=6),
                                                ],
                                                className="mb-4",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(dcc.Graph(id="crime-evolution-chart"), width=12, lg=6),
                                                    dbc.Col(dcc.Graph(id="weapon-types-chart"), width=12, lg=6),
                                                ],
                                                className="mb-4",
                                            ),
                                        ],
                                        label="Crime Type Analysis",
                                        tab_id="tab-crime-type",
                                    ),
                                    
                                    # Demographics Analysis
                                    dbc.Tab(
                                        [
                                            html.H4("Victim Demographics Analysis", className="mt-3 text-center"),
                                            html.P(
                                                "Explore crime patterns across different demographic groups.",
                                                className="text-center text-muted mb-4"
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(dcc.Graph(id="age-distribution-chart"), width=12, lg=6),
                                                    dbc.Col(dcc.Graph(id="gender-analysis-chart"), width=12, lg=6),
                                                ],
                                                className="mb-4",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(dcc.Graph(id="age-crime-heatmap"), width=12, lg=6),
                                                    dbc.Col(dcc.Graph(id="descent-analysis-chart"), width=12, lg=6),
                                                ],
                                                className="mb-4",
                                            ),
                                        ],
                                        label="Demographics Analysis",
                                        tab_id="tab-demographics",
                                    ),
                                    
                                    # Insights & Predictions
                                    dbc.Tab(
                                        [
                                            html.H4("Predictive Insights", className="mt-3 text-center"),
                                            html.P(
                                                "Explore patterns and predictions based on historical crime data.",
                                                className="text-center text-muted mb-4"
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(dcc.Graph(id="time-series-forecast"), width=12, lg=6),
                                                    dbc.Col(dcc.Graph(id="crime-correlation-heatmap"), width=12, lg=6),
                                                ],
                                                className="mb-4",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(dcc.Graph(id="crime-clustering"), width=12, lg=6),
                                                    dbc.Col(dcc.Graph(id="feature-importance-chart"), width=12, lg=6),
                                                ],
                                                className="mb-4",
                                            ),
                                        ],
                                        label="Insights & Predictions",
                                        tab_id="tab-insights",
                                    ),
                                    
                                    # Data Explorer
                                    dbc.Tab(
                                        [
                                            html.H4("Data Explorer", className="mt-3 text-center"),
                                            html.P(
                                                "Explore raw data with custom queries and filters.",
                                                className="text-center text-muted mb-4"
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.P("Records to display:"),
                                                            dcc.Slider(
                                                                id="records-slider",
                                                                min=100,
                                                                max=1000,
                                                                step=100,
                                                                value=500,
                                                                marks={100: "100", 500: "500", 1000: "1000"},
                                                                className="dark-slider"
                                                            ),
                                                        ],
                                                        width=12,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dash_table.DataTable(
                                                            id="data-table",
                                                            page_size=20,
                                                            style_table={'overflowX': 'auto'},
                                                            style_header={
                                                                'backgroundColor': 'rgb(30, 30, 30)',
                                                                'color': 'white',
                                                                'fontWeight': 'bold'
                                                            },
                                                            style_cell={
                                                                'backgroundColor': 'rgb(50, 50, 50)',
                                                                'color': 'white',
                                                                'textAlign': 'left',
                                                                'overflow': 'hidden',
                                                                'textOverflow': 'ellipsis',
                                                                'maxWidth': 0,
                                                            },
                                                            style_data_conditional=[
                                                                {
                                                                    'if': {'row_index': 'odd'},
                                                                    'backgroundColor': 'rgb(55, 55, 55)'
                                                                }
                                                            ],
                                                            export_format='csv',
                                                        ),
                                                        width=12,
                                                    ),
                                                ],
                                                className="mb-4",
                                            ),
                                        ],
                                        label="Data Explorer",
                                        tab_id="tab-data",
                                    ),
                                ],
                                id="dashboard-tabs",
                                active_tab="tab-temporal",
                            ),
                        ],
                        width=12, lg=9,
                    ),
                ]
            ),
            
            # Footer
            dbc.Row(
                dbc.Col(
                    html.P(
                        [
                            f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ", 
                            html.Span("Data source: LA City Open Data", className="text-muted"),
                        ],
                        className="text-center mt-4 mb-4"
                    ),
                    width=12
                )
            ),
        ],
        fluid=True,
        className="p-4",
    ),
])

# Filtering logic
def filter_dataframe(df, filters):
    filtered_df = df.copy()
    
    # Time filters
    if filters.get('year'):
        filtered_df = filtered_df[filtered_df["year"].isin(filters['year'])]
    if filters.get('season'):
        filtered_df = filtered_df[filtered_df["season"].isin(filters['season'])]
    if filters.get('time_of_day'):
        filtered_df = filtered_df[filtered_df["time_of_day"].isin(filters['time_of_day'])]
    
    # Location filters
    if filters.get('area'):
        filtered_df = filtered_df[filtered_df["area_name"].isin(filters['area'])]
    
    # Crime filters
    if filters.get('crime_category'):
        filtered_df = filtered_df[filtered_df["crime_category"].isin(filters['crime_category'])]
    if filters.get('crime'):
        filtered_df = filtered_df[filtered_df["crime_description"].isin(filters['crime'])]
    
    # Demographic filters
    if filters.get('victim_sex'):
        filtered_df = filtered_df[filtered_df["victim_sex_desc"].isin(filters['victim_sex'])]
    if filters.get('victim_descent'):
        filtered_df = filtered_df[filtered_df["victim_descent_desc"].isin(filters['victim_descent'])]
    if filters.get('age_group'):
        filtered_df = filtered_df[filtered_df["age_group"].isin(filters['age_group'])]
    
    return filtered_df

# Build filter store from all dropdowns
@callback(
    Output("filter-store", "data"),
    [
        Input("year-dropdown", "value"),
        Input("season-dropdown", "value"),
        Input("time-of-day-dropdown", "value"),
        Input("area-dropdown", "value"),
        Input("crime-category-dropdown", "value"),
        Input("crime-dropdown", "value"),
        Input("victim-sex-dropdown", "value"),
        Input("victim-descent-dropdown", "value"),
        Input("age-group-dropdown", "value"),
        Input("reset-button", "n_clicks")
    ]
)
# Use the cache to prevent unnecessary recalculations
@cache.memoize()
def update_filter_store(year, season, time_of_day, area, crime_category, crime, victim_sex, victim_descent, age_group, reset):
    from dash import ctx
    if not ctx.triggered:
        return {}
    
    trigger_id = ctx.triggered_id
    
    if trigger_id == "reset-button":
        return {}
    
    # Build the filters dictionary
    filters = {}
    if year:
        filters['year'] = year
    if season:
        filters['season'] = season
    if time_of_day:
        filters['time_of_day'] = time_of_day
    if area:
        filters['area'] = area
    if crime_category:
        filters['crime_category'] = crime_category
    if crime:
        filters['crime'] = crime
    if victim_sex:
        filters['victim_sex'] = victim_sex
    if victim_descent:
        filters['victim_descent'] = victim_descent
    if age_group:
        filters['age_group'] = age_group
    
    # Print for debugging
    print(f"Filters updated: {len(filters)} active filters")
    
    return filters

# Reset all dropdowns when button is clicked
@callback(
    [
        Output("year-dropdown", "value"),
        Output("season-dropdown", "value"),
        Output("time-of-day-dropdown", "value"),
        Output("area-dropdown", "value"),
        Output("crime-category-dropdown", "value"),
        Output("crime-dropdown", "value"),
        Output("victim-sex-dropdown", "value"),
        Output("victim-descent-dropdown", "value"),
        Output("age-group-dropdown", "value")
    ],
    Input("reset-button", "n_clicks"),
    prevent_initial_call=True
)
def reset_filters(n_clicks):
    return None, None, None, None, None, None, None, None, None

# KPI Cards
@callback(
    [
        Output("total-crimes-card", "children"),
        Output("most-common-crime-card", "children"),
        Output("most-dangerous-area-card", "children"),
        Output("time-pattern-card", "children")
    ],
    Input("filter-store", "data")
)
def update_kpi_cards(filters):
    if not filters:
        filters = {}
    
    filtered_df = filter_dataframe(df, filters)
    
    # Total crimes
    total_crimes = len(filtered_df)
    
    # Most common crime
    if not filtered_df.empty:
        crime_counts = filtered_df.groupby("crime_description").size()
        most_common_crime = crime_counts.idxmax()
        crime_percentage = (crime_counts.max() / total_crimes * 100)
    else:
        most_common_crime = "N/A"
        crime_percentage = 0
    
    # Most dangerous area
    if not filtered_df.empty:
        area_counts = filtered_df.groupby("area_name").size()
        most_dangerous_area = area_counts.idxmax()
        area_percentage = (area_counts.max() / total_crimes * 100)
    else:
        most_dangerous_area = "N/A"
        area_percentage = 0
    
    # Time pattern
    if not filtered_df.empty:
        time_counts = filtered_df.groupby("time_of_day").size()
        most_common_time = time_counts.idxmax()
        time_percentage = (time_counts.max() / total_crimes * 100)
    else:
        most_common_time = "N/A"
        time_percentage = 0
    
    # Create KPI cards
    total_crimes_card = [
        dbc.CardHeader("Total Crimes", className="text-center"),
        dbc.CardBody(
            [
                html.H3(f"{total_crimes:,}", className="card-title text-center"),
                html.P("Incidents", className="card-text text-center text-muted"),
            ]
        )
    ]
    
    most_common_crime_card = [
        dbc.CardHeader("Most Common Crime", className="text-center"),
        dbc.CardBody(
            [
                html.H5(f"{most_common_crime}", className="card-title text-center", style={"fontSize": "1.1rem"}),
                html.P(f"{crime_percentage:.1f}% of total", className="card-text text-center text-muted"),
            ]
        )
    ]
    
    most_dangerous_area_card = [
        dbc.CardHeader("Highest Crime Area", className="text-center"),
        dbc.CardBody(
            [
                html.H5(f"{most_dangerous_area}", className="card-title text-center", style={"fontSize": "1.1rem"}),
                html.P(f"{area_percentage:.1f}% of total", className="card-text text-center text-muted"),
            ]
        )
    ]
    
    time_pattern_card = [
        dbc.CardHeader("Most Common Time", className="text-center"),
        dbc.CardBody(
            [
                html.H5(f"{most_common_time}", className="card-title text-center", style={"fontSize": "1.1rem"}),
                html.P(f"{time_percentage:.1f}% of total", className="card-text text-center text-muted"),
            ]
        )
    ]
    
    return total_crimes_card, most_common_crime_card, most_dangerous_area_card, time_pattern_card

# Temporal Analysis Callbacks

@callback(
    Output("yearly-trend-line", "figure"),
    Input("filter-store", "data")
)
# Apply caching to expensive visualization functions
@cache.memoize()
def update_yearly_trend(filters):
    # For performance tracking
    start_time = time.time()
    
    if not filters:
        filters = {}
    
    filtered_df = filter_dataframe(df, filters)
    
    # Group by year and crime category
    yearly_counts = filtered_df.groupby(["year", "crime_category"]).size().reset_index(name="Count")
    
    fig = px.line(
        yearly_counts, 
        x="year", 
        y="Count",
        color="crime_category",
        markers=True,
        title="Yearly Crime Trend by Category",
        labels={"year": "Year", "Count": "Number of Crimes", "crime_category": "Crime Category"}
    )
    
    fig.update_layout(
        template="plotly_dark",
        xaxis=dict(tickmode="array", tickvals=years),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    # Print performance info
    print(f"Yearly trend chart generated in {time.time() - start_time:.2f} seconds")
    
    return fig

@callback(
    Output("monthly-trend-line", "figure"),
    Input("filter-store", "data")
)
def update_monthly_trend(filters):
    if not filters:
        filters = {}
    
    filtered_df = filter_dataframe(df, filters)
    
    # Use the year_month column already in the dataset
    # Convert to date for proper time series visualization
    monthly_counts = filtered_df.groupby("year_month").size().reset_index(name="Count")
    monthly_counts["year_month_date"] = pd.to_datetime(monthly_counts["year_month"] + "-01")
    monthly_counts = monthly_counts.sort_values("year_month_date")
    
    # Create figure
    fig = px.line(
        monthly_counts, 
        x="year_month_date", 
        y="Count",
        markers=True,
        title="Monthly Crime Trend",
        labels={"year_month_date": "Month", "Count": "Number of Crimes"}
    )
    
    # Add trendline
    if len(monthly_counts) > 1:
        x = np.array(range(len(monthly_counts)))
        y = monthly_counts["Count"].values
        model = LinearRegression().fit(x.reshape(-1, 1), y)
        y_pred = model.predict(x.reshape(-1, 1))
        
        fig.add_trace(
            go.Scatter(
                x=monthly_counts["year_month_date"],
                y=y_pred,
                mode="lines",
                line=dict(dash="dash", color="red"),
                name="Trend Line"
            )
        )
    
    fig.update_layout(
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

@callback(
    Output("hour-heatmap", "figure"),
    Input("filter-store", "data")
)
def update_hour_heatmap(filters):
    if not filters:
        filters = {}
    
    filtered_df = filter_dataframe(df, filters)
    
    # Create hour of day vs day of week heatmap
    hour_day_counts = filtered_df.groupby(["hour", "day_of_week"]).size().reset_index(name="Count")
    
    # Custom order for days of week
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    fig = px.density_heatmap(
        hour_day_counts,
        x="day_of_week",
        y="hour",
        z="Count",
        title="Crime by Hour and Day of Week",
        labels={"day_of_week": "Day of Week", "hour": "Hour of Day", "Count": "Number of Crimes"},
        color_continuous_scale="Viridis"
    )
    
    fig.update_layout(
        template="plotly_dark",
        xaxis={"categoryorder": "array", "categoryarray": day_order},
        yaxis={"categoryorder": "array", "categoryarray": list(range(24))},
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

@callback(
    Output("day-of-week-chart", "figure"),
    Input("filter-store", "data")
)
def update_day_of_week(filters):
    if not filters:
        filters = {}
    
    filtered_df = filter_dataframe(df, filters)
    
    # Count by day of week and crime category
    day_counts = filtered_df.groupby(["day_of_week", "crime_category"]).size().reset_index(name="Count")
    
    # Custom order for days of week
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    fig = px.bar(
        day_counts, 
        x="day_of_week", 
        y="Count",
        color="crime_category",
        title="Crimes by Day of Week and Category",
        labels={"day_of_week": "Day of Week", "Count": "Number of Crimes", "crime_category": "Crime Category"},
        barmode="group"
    )
    
    fig.update_layout(
        template="plotly_dark",
        xaxis={"categoryorder": "array", "categoryarray": day_order},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

@callback(
    Output("seasonal-pattern", "figure"),
    Input("filter-store", "data")
)
@cache.memoize()
def update_seasonal_pattern(filters):
    if not filters:
        filters = {}
    
    filtered_df = filter_dataframe(df, filters)
    
    # Group by season and year
    season_counts = filtered_df.groupby(["season", "year"]).size().reset_index(name="Count")
    
    # Define proper season order and create a numeric mapping for sorting
    season_order = ["Winter", "Spring", "Summer", "Fall"]
    season_map = {season: i for i, season in enumerate(season_order)}
    
    # Add a sort column based on the mapping
    season_counts["season_order"] = season_counts["season"].map(season_map)
    
    # Sort the dataframe by year and season_order
    season_counts = season_counts.sort_values(["year", "season_order"])
    
    # Create the figure with sorted data
    fig = px.line(
        season_counts, 
        x="season", 
        y="Count",
        color="year",
        title="Seasonal Crime Patterns",
        labels={"season": "Season", "Count": "Number of Crimes", "year": "Year"},
        markers=True,
        category_orders={"season": season_order}  # Explicitly set category order
    )
    
    fig.update_layout(
        template="plotly_dark",
        xaxis={"categoryorder": "array", "categoryarray": season_order},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

# Spatial Analysis Callbacks

@callback(
    Output("crime-heatmap", "figure"),
    [Input("filter-store", "data"), 
     Input("sample-slider", "value")]
)
# Apply caching to the most expensive visualization
@cache.memoize()
def update_crime_heatmap(filters, sample_size):
    # Track performance
    start_time = time.time()
    
    if not filters:
        filters = {}
    
    filtered_df = filter_dataframe(df, filters)
    
    # Limit sample size for performance
    if sample_size > 20000:
        print(f"WARNING: Limiting sample size from {sample_size} to 20000 for performance")
        sample_size = 20000
    
    # Take a sample for faster rendering
    if len(filtered_df) > sample_size:
        filtered_df = filtered_df.sample(sample_size, random_state=42)
    
    print(f"Generating heatmap with {len(filtered_df)} points...")
    
    try:
        # Create a heatmap focusing on the density of crimes
        fig = px.density_mapbox(
            filtered_df,
            lat="latitude",
            lon="longitude",
            z=None,  # We'll let density_mapbox calculate point density
            radius=10,
            center={"lat": 34.0522, "lon": -118.2437},
            zoom=9,
            mapbox_style="carto-darkmatter",
            title="Crime Heatmap",
            color_continuous_scale="Viridis",
            opacity=0.7,
            hover_data=["crime_description", "date_occurred", "area_name"]
        )
        
        fig.update_layout(
            template="plotly_dark",
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
        )
        
        print(f"Heatmap generated in {time.time() - start_time:.2f} seconds")
        return fig
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        # Return an empty figure with an error message
        fig = go.Figure()
        fig.update_layout(
            title="Error generating heatmap - please try with a smaller sample size",
            template="plotly_dark"
        )
        return fig

@callback(
    Output("area-comparison-chart", "figure"),
    Input("filter-store", "data")
)
def update_area_comparison(filters):
    if not filters:
        filters = {}
    
    filtered_df = filter_dataframe(df, filters)
    
    area_counts = filtered_df.groupby("area_name").size().reset_index(name="Count")
    area_counts = area_counts.sort_values("Count", ascending=False).head(10)
    
    fig = px.bar(
        area_counts, 
        x="Count", 
        y="area_name",
        orientation="h",
        title="Top 10 Areas by Crime Count",
        labels={"area_name": "Area", "Count": "Number of Crimes"},
        color="Count",
        color_continuous_scale="Viridis"
    )
    
    fig.update_layout(
        template="plotly_dark",
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

@callback(
    Output("crime-by-area-pie", "figure"),
    Input("filter-store", "data")
)
def update_crime_by_area_pie(filters):
    if not filters:
        filters = {}
    
    filtered_df = filter_dataframe(df, filters)
    
    # Get crime category distribution for selected area(s)
    if filters.get('area') and len(filters['area']) == 1:
        # If only one area is selected, show distribution by crime category
        title = f"Crime Categories in {filters['area'][0]}"
        category_counts = filtered_df.groupby("crime_category").size().reset_index(name="Count")
        fig = px.pie(
            category_counts,
            values="Count",
            names="crime_category",
            title=title,
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Viridis
        )
    else:
        # Otherwise show top 6 areas
        title = "Top 6 Areas by Crime Count"
        area_counts = filtered_df.groupby("area_name").size().reset_index(name="Count")
        area_counts = area_counts.sort_values("Count", ascending=False).head(6)
        fig = px.pie(
            area_counts,
            values="Count",
            names="area_name",
            title=title,
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Viridis
        )
    
    fig.update_layout(
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

# Crime Type Analysis Callbacks

@callback(
    Output("crime-treemap", "figure"),
    Input("filter-store", "data")
)
def update_crime_treemap(filters):
    if not filters:
        filters = {}
    
    filtered_df = filter_dataframe(df, filters)
    
    # Create hierarchical data for treemap (crime category -> crime description)
    crime_hierarchy = filtered_df.groupby(["crime_category", "crime_description"]).size().reset_index(name="Count")
    
    fig = px.treemap(
        crime_hierarchy,
        path=["crime_category", "crime_description"],
        values="Count",
        title="Crime Categories and Types",
        color="Count",
        color_continuous_scale="Viridis"
    )
    
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

@callback(
    Output("top-crimes-chart", "figure"),
    Input("filter-store", "data")
)
def update_top_crimes(filters):
    if not filters:
        filters = {}
    
    filtered_df = filter_dataframe(df, filters)
    
    # Count top crime types
    crime_counts = filtered_df.groupby("crime_description").size().reset_index(name="Count")
    crime_counts = crime_counts.sort_values("Count", ascending=False).head(10)
    
    fig = px.bar(
        crime_counts, 
        x="Count", 
        y="crime_description",
        orientation="h",
        title="Top 10 Crime Types",
        labels={"crime_description": "Crime Type", "Count": "Number of Crimes"},
        color="Count",
        color_continuous_scale="Viridis"
    )
    
    fig.update_layout(
        template="plotly_dark",
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

@callback(
    Output("crime-evolution-chart", "figure"),
    Input("filter-store", "data")
)
def update_crime_evolution(filters):
    if not filters:
        filters = {}
    
    filtered_df = filter_dataframe(df, filters)
    
    # For crime evolution, we need to compare across time periods
    # Group by year-month and crime category
    evolution_df = filtered_df.copy()
    
    # Use the year_month column directly
    # Convert to date for time series visualization
    evolution_df["year_month_date"] = pd.to_datetime(evolution_df["year_month"] + "-01")
    
    # Get top 5 crime categories for readability
    top_categories = evolution_df.groupby("crime_category").size().nlargest(5).index.tolist()
    evolution_df = evolution_df[evolution_df["crime_category"].isin(top_categories)]
    
    # Group by year-month and crime category
    crime_evolution = evolution_df.groupby(["year_month_date", "crime_category"]).size().reset_index(name="Count")
    
    fig = px.line(
        crime_evolution,
        x="year_month_date",
        y="Count",
        color="crime_category",
        title="Evolution of Top 5 Crime Categories Over Time",
        labels={"year_month_date": "Date", "Count": "Number of Crimes", "crime_category": "Crime Category"},
        markers=True
    )
    
    fig.update_layout(
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

@callback(
    Output("weapon-types-chart", "figure"),
    Input("filter-store", "data")
)
def update_weapon_types(filters):
    if not filters:
        filters = {}
    
    filtered_df = filter_dataframe(df, filters)
    
    # Count weapon types, handle missing values
    filtered_df_weapons = filtered_df.dropna(subset=["weapon_description"])
    filtered_df_weapons = filtered_df_weapons[filtered_df_weapons["weapon_description"] != "UNKNOWN"]
    
    # Skip if no weapon data
    if len(filtered_df_weapons) == 0:
        fig = go.Figure()
        fig.update_layout(
            title="No Weapon Data Available",
            template="plotly_dark"
        )
        return fig
    
    weapon_counts = filtered_df_weapons.groupby("weapon_description").size().reset_index(name="Count")
    weapon_counts = weapon_counts.sort_values("Count", ascending=False).head(10)
    
    fig = px.pie(
        weapon_counts,
        values="Count",
        names="weapon_description",
        title="Top Weapon Types Used",
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    
    fig.update_layout(
        template="plotly_dark",
        legend_title="Weapon Type",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

# Demographics Analysis Callbacks

@callback(
    Output("age-distribution-chart", "figure"),
    Input("filter-store", "data")
)
def update_age_distribution(filters):
    if not filters:
        filters = {}
    
    filtered_df = filter_dataframe(df, filters)
    
    # Filter out extreme ages for better visualization
    age_df = filtered_df[(filtered_df["victim_age"] > 0) & (filtered_df["victim_age"] < 100)]
    
    fig = px.histogram(
        age_df,
        x="victim_age",
        color="victim_sex_desc",
        title="Victim Age Distribution",
        labels={"victim_age": "Age", "count": "Number of Victims", "victim_sex_desc": "Gender"},
        nbins=20,
        opacity=0.7,
        barmode="overlay"
    )
    
    fig.update_layout(
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

@callback(
    Output("gender-analysis-chart", "figure"),
    Input("filter-store", "data")
)
def update_gender_analysis(filters):
    if not filters:
        filters = {}
    
    filtered_df = filter_dataframe(df, filters)
    
    # Group by crime category and victim sex
    gender_crime = filtered_df.groupby(["crime_category", "victim_sex_desc"]).size().reset_index(name="Count")
    gender_crime = gender_crime[gender_crime["victim_sex_desc"] != "Unknown"]
    
    fig = px.bar(
        gender_crime,
        x="crime_category",
        y="Count",
        color="victim_sex_desc",
        title="Crime Types by Victim Gender",
        labels={"crime_category": "Crime Category", "Count": "Number of Crimes", "victim_sex_desc": "Gender"},
        barmode="group"
    )
    
    fig.update_layout(
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

@callback(
    Output("age-crime-heatmap", "figure"),
    Input("filter-store", "data")
)
def update_age_crime_heatmap(filters):
    if not filters:
        filters = {}
    
    filtered_df = filter_dataframe(df, filters)
    
    # Create heatmap of age group vs crime category
    age_crime = filtered_df.groupby(["age_group", "crime_category"]).size().reset_index(name="Count")
    
    # Order for age groups
    age_order = ["Under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+", "Unknown"]
    
    fig = px.density_heatmap(
        age_crime,
        x="crime_category",
        y="age_group",
        z="Count",
        title="Crime Categories by Victim Age Group",
        labels={"crime_category": "Crime Category", "age_group": "Age Group", "Count": "Number of Crimes"},
        color_continuous_scale="Viridis"
    )
    
    fig.update_layout(
        template="plotly_dark",
        yaxis={"categoryorder": "array", "categoryarray": age_order},
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

@callback(
    Output("descent-analysis-chart", "figure"),
    Input("filter-store", "data")
)
def update_descent_analysis(filters):
    if not filters:
        filters = {}
    
    filtered_df = filter_dataframe(df, filters)
    
    # Get top 6 descent groups for cleaner visualization
    top_descents = filtered_df.groupby("victim_descent_desc").size().nlargest(6).index.tolist()
    descent_df = filtered_df[filtered_df["victim_descent_desc"].isin(top_descents)]
    
    # Group by crime category and victim descent
    descent_crime = descent_df.groupby(["crime_category", "victim_descent_desc"]).size().reset_index(name="Count")
    
    fig = px.bar(
        descent_crime,
        x="crime_category",
        y="Count",
        color="victim_descent_desc",
        title="Crime Categories by Victim Descent",
        labels={"crime_category": "Crime Category", "Count": "Number of Crimes", "victim_descent_desc": "Descent"},
        barmode="group"
    )
    
    fig.update_layout(
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

# Insights & Predictions Callbacks

@callback(
    Output("time-series-forecast", "figure"),
    Input("filter-store", "data")
)
def update_time_series_forecast(filters):
    if not filters:
        filters = {}
    
    # For forecasting, we'll use monthly data regardless of filters
    # But we'll apply crime type and area filters
    forecast_df = df.copy()
    
    if filters.get('crime'):
        forecast_df = forecast_df[forecast_df["crime_description"].isin(filters['crime'])]
    if filters.get('crime_category'):
        forecast_df = forecast_df[forecast_df["crime_category"].isin(filters['crime_category'])]
    if filters.get('area'):
        forecast_df = forecast_df[forecast_df["area_name"].isin(filters['area'])]
    
    # Use year_month already in the dataset
    # Convert to date for proper time series visualization
    monthly_counts = forecast_df.groupby("year_month").size().reset_index(name="Count")
    monthly_counts["year_month_date"] = pd.to_datetime(monthly_counts["year_month"] + "-01")
    monthly_counts = monthly_counts.sort_values("year_month_date")
    
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=monthly_counts["year_month_date"],
            y=monthly_counts["Count"],
            mode="lines+markers",
            name="Historical Data",
            line=dict(color="#636EFA")
        )
    )
    
    # Add trend and forecast if we have enough data
    if len(monthly_counts) > 6:
        # Simple linear regression model for trend
        X = np.array(range(len(monthly_counts))).reshape(-1, 1)
        y = monthly_counts["Count"].values
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        
        # Extend the forecast for 6 months
        X_future = np.array(range(len(monthly_counts), len(monthly_counts) + 6)).reshape(-1, 1)
        y_future = model.predict(X_future)
        
        # Calculate future dates
        last_date = monthly_counts["year_month_date"].iloc[-1]
        future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(6)]
        
        # Add trend line for historical data
        fig.add_trace(
            go.Scatter(
                x=monthly_counts["year_month_date"],
                y=y_pred,
                mode="lines",
                line=dict(dash="dash", color="#EF553B"),
                name="Trend Line"
            )
        )
        
        # Add forecast for future data
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=y_future,
                mode="lines+markers",
                line=dict(dash="dash", color="#EF553B"),
                marker=dict(color="#EF553B"),
                name="Forecast (6 months)"
            )
        )
        
        # Add confidence interval (simple approach for illustration)
        y_err = np.std(y - y_pred)
        y_upper = y_future + 1.96 * y_err
        y_lower = y_future - 1.96 * y_err
        
        fig.add_trace(
            go.Scatter(
                x=future_dates + future_dates[::-1],
                y=np.concatenate([y_upper, y_lower[::-1]]),
                fill="toself",
                fillcolor="rgba(231, 107, 243, 0.2)",
                line=dict(color="rgba(255, 255, 255, 0)"),
                name="95% Confidence Interval"
            )
        )
    
    fig.update_layout(
        template="plotly_dark",
        title="Time Series Forecast (Next 6 Months)",
        xaxis_title="Date",
        yaxis_title="Number of Crimes",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

@callback(
    Output("crime-correlation-heatmap", "figure"),
    Input("filter-store", "data")
)
def update_correlation_heatmap(filters):
    if not filters:
        filters = {}
    
    filtered_df = filter_dataframe(df, filters)
    
    # Create correlation data
    # We'll look at crime counts by hour, day of week, month, etc.
    corr_df = filtered_df.copy()
    
    # Aggregate by various time dimensions
    hour_counts = corr_df.groupby("hour").size().reset_index(name="hour_count")
    day_counts = corr_df.groupby("day").size().reset_index(name="day_count")
    month_counts = corr_df.groupby("month").size().reset_index(name="month_count")
    
    # Create weekday counts with numeric values for day of week
    # First create the counts
    weekday_counts = corr_df.groupby("day_of_week").size().reset_index(name="weekday_count")
    # Then map the weekday names to numbers
    day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    weekday_counts["day_number"] = weekday_counts["day_of_week"].map(day_mapping)
    
    # Create a correlation matrix with the counts
    corr_data = pd.DataFrame({
        "Hour": np.correlate(hour_counts["hour"], hour_counts["hour_count"], mode="same"),
        "Day": np.correlate(day_counts["day"], day_counts["day_count"], mode="same")[:30],
        "Month": np.correlate(month_counts["month"], month_counts["month_count"], mode="same")[:30],
    })
    
    # Normalize and create correlation matrix
    corr_data = corr_data.corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_data,
        title="Correlation Between Time Dimensions and Crime Occurrences",
        color_continuous_scale="Viridis"
    )
    
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

@callback(
    Output("crime-clustering", "figure"),
    Input("filter-store", "data")
)
def update_crime_clustering(filters):
    if not filters:
        filters = {}
    
    filtered_df = filter_dataframe(df, filters)
    
    # Create clustering based on time and location
    # We'll use coordinates, hour, and day of week
    cluster_df = filtered_df.dropna(subset=["latitude", "longitude", "hour"]).sample(min(5000, len(filtered_df)), random_state=42)
    
    # Ensure we have enough samples for clustering
    min_samples_needed = 5  # We need at least 5 samples for 5 clusters
    if len(cluster_df) < min_samples_needed:
        # Not enough data, create a simple cluster
        cluster_df["cluster"] = 0  # Assign all to one cluster
    else:
        # Extract features for clustering
        X = cluster_df[["latitude", "longitude", "hour"]].copy()
        
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine the number of clusters based on data size
        n_clusters = min(5, len(cluster_df))
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add clusters to dataframe
        cluster_df["cluster"] = clusters
    
    # Create scatter plot
    fig = px.scatter_mapbox(
        cluster_df,
        lat="latitude",
        lon="longitude",
        color="cluster",
        title="Crime Clusters Based on Location and Time",
        hover_data=["crime_description", "hour", "day_of_week"],
        color_continuous_scale="Viridis",
        zoom=9
    )
    
    fig.update_layout(
        template="plotly_dark",
        mapbox_style="carto-darkmatter",
        mapbox_center={"lat": 34.0522, "lon": -118.2437},
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    
    return fig

@callback(
    Output("feature-importance-chart", "figure"),
    Input("filter-store", "data")
)
def update_feature_importance(filters):
    if not filters:
        filters = {}
    
    filtered_df = filter_dataframe(df, filters)
    
    # Extract features for a simple model
    # We'll use this to illustrate a feature importance chart
    # In a real implementation, this would be a more sophisticated model
    
    # For simplicity, we'll predict crime category from time and location features
    model_df = filtered_df.dropna(subset=["hour", "day", "month", "latitude", "longitude"]).sample(min(10000, len(filtered_df)), random_state=42)
    
    # Create dummy variables for crime category
    crime_cat_dummy = pd.get_dummies(model_df["crime_category"])
    
    # Features
    features = ["hour", "day", "month", "latitude", "longitude"]
    X = model_df[features]
    y = (model_df["crime_category"] == "Violent Crime").astype(int)  # Predicting violent crimes
    
    # Train a simple model
    model = LinearRegression()
    model.fit(X, y)
    
    # Create a coefficients dataframe
    coef_df = pd.DataFrame({
        "Feature": features,
        "Importance": np.abs(model.coef_)
    })
    coef_df = coef_df.sort_values("Importance", ascending=False)
    
    # Create a bar chart
    fig = px.bar(
        coef_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance for Predicting Violent Crimes",
        labels={"Importance": "Absolute Coefficient Magnitude", "Feature": "Feature"},
        color="Importance",
        color_continuous_scale="Viridis"
    )
    
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

# Data Explorer Callback
@callback(
    Output("data-table", "data"),
    Output("data-table", "columns"),
    [Input("filter-store", "data"),
     Input("records-slider", "value")]
)
def update_data_table(filters, record_count):
    if not filters:
        filters = {}
    
    filtered_df = filter_dataframe(df, filters)
    
    # Sample data for performance
    if len(filtered_df) > record_count:
        filtered_df = filtered_df.sample(record_count, random_state=42)
    
    # Select columns to display
    display_columns = [
        "division_number", "date_occurred", "time_occurred", "area_name", 
        "crime_description", "victim_sex_desc", "victim_descent_desc",
        "victim_age", "weapon_description", "premise_description"
    ]
    
    # Format datetime columns
    table_df = filtered_df[display_columns].copy()
    if "date_occurred" in table_df.columns:
        table_df["date_occurred"] = table_df["date_occurred"].dt.strftime('%Y-%m-%d')
    
    # Create columns configuration
    columns = [{"name": col.replace("_", " ").title(), "id": col} for col in display_columns]
    
    return table_df.to_dict('records'), columns

# Add a timestamp route to verify the server is running
@server.route('/status')
def get_status():
    return f"Server running. Data loaded at {DATA_LOAD_TIME}"

# Run the app with modified settings
if __name__ == "__main__":
    print("Starting server...")
    # Set longer timeout and disable hot reload to prevent infinite reloading
    app.run_server(debug=False, dev_tools_hot_reload=False, threaded=True)