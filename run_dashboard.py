#!/usr/bin/env python3
"""
Launcher script for the LA Crime Dashboard
This script ensures that data is loaded properly and handles any startup issues.
"""
import os
import sys
import traceback
import time

def print_section(title):
    """Print a section heading"""
    print(f"\n{'=' * 50}")
    print(f" {title}")
    print(f"{'=' * 50}")

def create_data_directory():
    """Make sure data directory exists"""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    if not os.path.exists(data_dir):
        print(f"Creating data directory: {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
    return data_dir

def check_requirements():
    """Check if required packages are installed"""
    print_section("Checking Requirements")
    try:
        import pandas as pd
        import numpy as np
        import plotly
        import dash
        import dash_bootstrap_components as dbc
        from flask_caching import Cache
        import sklearn
        
        print("All required packages found!")
        return True
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("\nPlease install all requirements with:")
        print("pip install -r requirements.txt")
        return False

def check_data_files():
    """Check if sample data exists, generate if needed"""
    print_section("Checking Data Files")
    data_dir = create_data_directory()
    sample_file = os.path.join(data_dir, 'Crime_Data_from_2020_to_Present2025.csv')
    
    if os.path.exists(sample_file):
        print(f"Sample data found: {sample_file}")
        return True
    
    print("Sample data not found. Would you like to generate sample data? (y/n)")
    response = input("> ").strip().lower()
    
    if response == 'y':
        print("Generating sample data...")
        try:
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generate_sample_data.py')
            if os.path.exists(script_path):
                # Try to run the generator script
                print(f"Running: {script_path}")
                os.system(f"{sys.executable} {script_path}")
                
                if os.path.exists(sample_file):
                    print("Sample data generated successfully!")
                    return True
                else:
                    print("Error: Sample data file not created.")
                    return False
            else:
                print(f"Error: Could not find generator script at {script_path}")
                return False
        except Exception as e:
            print(f"Error generating sample data: {e}")
            traceback.print_exc()
            return False
    else:
        print("Continuing without sample data. Dashboard may not work properly.")
        return False

def launch_dashboard():
    """Launch the dashboard application"""
    print_section("Launching Dashboard")
    
    try:
        # Launch the dashboard with lower worker timeout
        print("Starting dashboard...")
        dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard.py')
        
        # Run with modified server settings
        cmd = f"{sys.executable} {dashboard_path}"
        os.system(cmd)
        
        return True
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        traceback.print_exc()
        return False

def main():
    """Main entry point"""
    print_section("LA Crime Dashboard Launcher")
    
    # Check requirements first
    if not check_requirements():
        print("Exiting due to missing requirements.")
        return 1
    
    # Check data files
    check_data_files()
    
    # Launch the dashboard
    if not launch_dashboard():
        print("Failed to launch dashboard.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)