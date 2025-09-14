import time
import os
import json
import pandas as pd
from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException, TimeoutException

# Add data_process to path for imports
import sys
sys.path.append('/Users/noamcohen/PycharmProjects/TrafCast/data_process')

from process import prepare_data_df, build_sensor_index, map_pms_to_sensors

# Paths
download_path = '/Users/noamcohen/Downloads/data collection/data'
roads_dir = '/Users/noamcohen/PycharmProjects/TrafCast/data/Los Angeles/roads'
metadata_file = os.path.join(roads_dir, 'road_metadata.json')
coordinates_dir = '/Users/noamcohen/PycharmProjects/TrafCast/data collection/coordinates'


def create_chrome_driver():
    """Create a new Chrome WebDriver instance with proper options."""
    chrome_options = Options()
    chrome_prefs = {
        "download.default_directory": download_path,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    chrome_options.add_experimental_option("prefs", chrome_prefs)
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    return webdriver.Chrome(service=Service('/opt/homebrew/bin/chromedriver'), options=chrome_options)


def login_to_pems(driver):
    """Login to PeMS website."""
    print("Logging into PeMS...")
    url = "https://pems.dot.ca.gov/"
    driver.get(url)
    time.sleep(20)
    
    # Find and fill login fields
    username_field = driver.find_element(By.ID, "username")
    password_field = driver.find_element(By.ID, "password")
    
    username_field.send_keys("amitomer1912@gmail.com")
    password_field.send_keys("5^applel?X")
    
    # Click login button
    login_button = driver.find_element(By.NAME, "login")
    login_button.click()
    time.sleep(20)
    print("Login completed")

def load_metadata():
    """Load road metadata from JSON file."""
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return {}

def save_metadata(metadata):
    """Save road metadata to JSON file."""
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

def normalize_date(date_input):
    """Normalize date to YYYY-MM-DD format."""
    if isinstance(date_input, str):
        # Try different date formats
        for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%m*%d*%Y"]:
            try:
                dt = datetime.strptime(date_input, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        return date_input  # Return as-is if no format matches
    elif isinstance(date_input, datetime):
        return date_input.strftime("%Y-%m-%d")
    return str(date_input)

def check_date_exists(road_key, target_date):
    """Check if a date already exists for a road in metadata."""
    metadata = load_metadata()
    if road_key not in metadata:
        return False

    # Normalize the target date
    normalized_target = normalize_date(target_date)

    # Check against normalized stored dates
    stored_dates = metadata[road_key].get('collected_dates', [])
    normalized_stored = [normalize_date(date) for date in stored_dates]

    return normalized_target in normalized_stored


def extract_time_params(target_datetime):
    """
    Extract day and s_time_id from datetime for PeMS URL.
    
    Args:
        target_datetime: datetime object or string in format "YYYY-MM-DD"
    
    Returns:
        tuple: (day, s_time_id, formatted_date)
    """
    if isinstance(target_datetime, str):
        target_datetime = datetime.strptime(target_datetime, "%Y-%m-%d")
    
    # Extract day number
    day = target_datetime.day
    
    # Calculate s_time_id (Unix timestamp)
    # PeMS uses Unix timestamp for the date
    s_time_id = int(target_datetime.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    
    # Format date for URL (MM/DD/YYYY)
    formatted_date = target_datetime.strftime("%m/%d/%Y")
    
    return day, s_time_id, formatted_date


def download_road_data(driver, road, direction, target_date, max_retries=5):
    """
    Download data for a specific road, direction, and date.
    
    Args:
        driver: WebDriver instance
        road: Road number (e.g., '405', '101')
        direction: Direction ('N', 'S', 'E', 'W')
        target_date: Date string "YYYY-MM-DD" or datetime object
        max_retries: Number of retry attempts
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Starting download for {road} {direction} on {target_date}")
    
    # Extract time parameters
    day, s_time_id, formatted_date = extract_time_params(target_date)
    day_str = f"0{day}" if day <= 9 else str(day)
    
    # Generate report URL  
    report_url = f"https://pems.dot.ca.gov/?report_form=1&dnode=Freeway&content=spatial&tab=contours&export=&fwy={road}&dir={direction}&s_time_id={s_time_id}&s_time_id_f={formatted_date.replace('/', '%2F')}&from_hh=0&to_hh=23&start_pm=.0&end_pm=1000.09&lanes=&station_type=ml&q=speed&colormap=30%2C31%2C32&sc=auto&ymin=&ymax=&view_d=2&chart.x=93&chart.y=20"
    
    for attempt in range(max_retries):
        try:
            print(f"Download attempt {attempt + 1}/{max_retries} for {road}_{direction}_{day_str}")
            
            # Navigate to report page
            driver.get(report_url)
            time.sleep(60)
            
            # Find and click export button
            export_button = driver.find_element(By.NAME, "xls")
            export_button.click()
            
            # Wait for download
            time.sleep(100)
            
            # Process the downloaded file (rename and move)
            file_success = process_downloaded_file(road, direction, formatted_date, day_str)

            if file_success:
                # Also process and append to CSV files
                date_safe = formatted_date.replace("/", "*")
                temp_file_name = f"{road}_{direction}_{date_safe}.xlsx"
                road_dir = os.path.join(download_path, str(road), direction)
                temp_file_path = os.path.join(road_dir, temp_file_name)

                # Convert road format for process_and_append_data
                road_key = convert_to_metadata_key(road, direction)
                data_success = process_and_append_data(temp_file_path, road_key, direction, target_date)

                if data_success:
                    print(f"Successfully downloaded and processed {road}_{direction}_{day_str}")
                    return True
                else:
                    raise Exception("Data processing to CSV failed")
            else:
                raise Exception("File processing failed")
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {road}_{direction}_{day_str}: {e}")
            if attempt < max_retries - 1:
                # Progressive backoff
                backoff_time = (attempt + 1) * 30
                print(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
            else:
                print(f"Failed to download {road}_{direction}_{day_str} after {max_retries} attempts")
                return False
    
    return False


def process_downloaded_file(road, direction, formatted_date, day_str):
    """Process and rename the downloaded file."""
    try:
        # Fix for Python 3.11 f-string nesting issue
        date_safe = formatted_date.replace("/", "*")
        new_file_name = f"{road}_{direction}_{date_safe}.xlsx"
        
        # Create road-specific directory
        road_dir = os.path.join(download_path, str(road), direction)
        os.makedirs(road_dir, exist_ok=True)
        final_file_path = os.path.join(road_dir, new_file_name)
        
        # Original file path
        original_file_path = os.path.join(download_path, 'pems_output.xlsx')
        
        # Check for file existence and rename
        if os.path.exists(original_file_path):
            os.rename(original_file_path, final_file_path)
            print(f"File saved to: {final_file_path}")
            return True
        else:
            print("Original file not found!")
            return False
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

def get_coordinate_file(road_key, direction):
    """Get the coordinate file path for a road and direction.

    Args:
        road_key: Metadata key format like 'I_405_north', 'US_101_south', 'CA_134_east'
        direction: Direction like 'N', 'S', 'E', 'W' (not used, parsed from road_key)
    """
    # Parse the metadata key: "I_405_north" -> ["I", "405", "north"]
    parts = road_key.split('_')
    if len(parts) != 3:
        print(f"Invalid road_key format: {road_key}")
        return None

    prefix, road_num, direction_full = parts

    # Convert to coordinate file format: "I 405 North"
    coord_road = f"{prefix} {road_num}"
    direction_title = direction_full.title()  # north -> North

    coord_filename = f"{coord_road} {direction_title}.xlsx"
    coord_path = os.path.join(coordinates_dir, coord_filename)

    if os.path.exists(coord_path):
        return coord_path
    else:
        print(f"Coordinate file not found: {coord_path}")
        return None

def process_and_append_data(temp_file, road_name, direction, target_date):
    """Process downloaded Excel using unified.py logic and append to CSV."""
    try:
        # Read downloaded Excel file
        df_data = pd.read_excel(temp_file)
        
        if df_data.empty:
            print("Downloaded file is empty")
            return False
        
        # Get coordinate file
        coord_path = get_coordinate_file(road_name, direction)
        if not coord_path:
            print(f"No coordinate file found for {road_name} {direction}")
            return False
        
        df_coord = pd.read_excel(coord_path)
        
        # Process data using the same logic as unified.py
        clean_df = prepare_data_df(df_data, df_coord, target_date)
        if clean_df.empty:
            print("No data after cleaning")
            return False
            
        sensors = build_sensor_index(clean_df)
        enriched = map_pms_to_sensors(clean_df, sensors)
        
        # Add road name and direction to match existing CSV format
        enriched['road_name'] = road_name.replace('_', ' ').upper()
        enriched['direction'] = direction.title()

        # Use road_name directly as it's already in correct format (I_405_north)
        csv_file = os.path.join(roads_dir, f"{road_name}.csv.gz")
        
        # Check if file exists and handle corrupted/empty files
        if os.path.exists(csv_file):
            try:
                # Try to read existing compressed CSV
                existing_df = pd.read_csv(csv_file, compression='gzip', nrows=1)

                # If file has data, append to it
                if not existing_df.empty:
                    # Ensure new data has same columns as existing
                    for col in existing_df.columns:
                        if col not in enriched.columns:
                            enriched[col] = None

                    # Reorder columns to match existing CSV
                    enriched = enriched[existing_df.columns]

                    # Read existing data, append new data, and save back compressed
                    existing_full = pd.read_csv(csv_file, compression='gzip')
                    combined_df = pd.concat([existing_full, enriched], ignore_index=True)
                    combined_df.to_csv(csv_file, compression='gzip', index=False)
                    print(f"Appended {len(enriched)} rows to existing compressed {csv_file}")
                else:
                    # File is empty, treat as new file
                    enriched.to_csv(csv_file, compression='gzip', index=False)
                    print(f"Overwrote empty file {csv_file} with {len(enriched)} rows")

            except (pd.errors.EmptyDataError, EOFError, Exception) as e:
                # File is corrupted or not properly compressed, overwrite it
                print(f"Warning: Existing file corrupted ({e}), overwriting...")
                enriched.to_csv(csv_file, compression='gzip', index=False)
                print(f"Overwrote corrupted {csv_file} with {len(enriched)} rows")
        else:
            # Create new compressed CSV file
            enriched.to_csv(csv_file, compression='gzip', index=False)
            print(f"Created new compressed {csv_file} with {len(enriched)} rows")
        
        # Update metadata
        update_metadata(road_name, target_date)
        
        return True
        
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return False

def update_metadata(road_key, date_str):
    """Update metadata to mark date as collected."""
    metadata = load_metadata()

    # Normalize the date before storing
    normalized_date = normalize_date(date_str)

    if road_key not in metadata:
        metadata[road_key] = {
            'csv_file': f"{road_key}.csv",
            'collected_dates': [],
            'last_updated': datetime.now().isoformat()
        }

    # Check if normalized date already exists
    existing_normalized = [normalize_date(date) for date in metadata[road_key]['collected_dates']]

    if normalized_date not in existing_normalized:
        metadata[road_key]['collected_dates'].append(normalized_date)
        metadata[road_key]['collected_dates'].sort()  # Keep dates sorted

    metadata[road_key]['last_updated'] = datetime.now().isoformat()
    save_metadata(metadata)
    print(f"Updated metadata: {road_key} now has date {normalized_date}")

def convert_to_metadata_key(road_name, direction):
    """Convert road name and direction to metadata key format.

    Args:
        road_name: Road number like '405', '101', '134'
        direction: Direction like 'N', 'S', 'E', 'W'

    Returns:
        str: Metadata key like 'I_405_north', 'US_101_south', 'CA_134_east'
    """
    # Convert direction to full name
    direction_map = {'N': 'north', 'S': 'south', 'E': 'east', 'W': 'west'}
    direction_full = direction_map.get(direction.upper(), direction.lower())

    # Determine road prefix based on common LA highways
    if road_name in ['405', '5', '110', '210', '605']:
        prefix = 'I'
    elif road_name in ['101']:
        prefix = 'US'
    elif road_name in ['134', '118', '170']:
        prefix = 'CA'
    else:
        # Default to I for Interstate
        prefix = 'I'

    return f"{prefix}_{road_name}_{direction_full}"

def collect_road_data_if_missing(road_name, direction, target_date):
    """
    Check if data exists for a road/date, download and append if missing.

    Args:
        road_name: Road name like '405', '134', '101' (just the number)
        direction: Direction like 'N', 'S', 'E', 'W' (single letter)
        target_date: Date string 'YYYY-MM-DD'

    Returns:
        str: 'exists', 'success', or 'failed'
    """
    # Convert to metadata format: I_405_north, US_101_south, etc.
    road_key = convert_to_metadata_key(road_name, direction)

    # Check if date already exists BEFORE any browser operations
    if check_date_exists(road_key, target_date):
        print(f"{road_key} already has data for {target_date}")
        return 'exists'

    print(f"{road_key} missing data for {target_date}, downloading...")

    driver = None
    try:
        # Create driver and login
        driver = create_chrome_driver()
        login_to_pems(driver)

        # Download and process data
        success = download_road_data(driver, road_name, direction, target_date)

        return 'success' if success else 'failed'

    except Exception as e:
        print(f"Error: {e}")
        return 'failed'

    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass


def collect_roads_data(roads_list, target_date):
    """
    Download data for multiple roads and their directions on a specific date.

    Args:
        roads_list: List of tuples [(road, direction), ...] e.g., [('405', 'N'), ('101', 'S')]
        target_date: Date string "YYYY-MM-DD" or datetime object

    Returns:
        dict: Results dictionary with success/failure status for each road
    """
    # Create download directory
    os.makedirs(download_path, exist_ok=True)

    results = {}

    # First, check which roads actually need downloading BEFORE browser operations
    roads_to_download = []
    for road, direction in roads_list:
        road_key = convert_to_metadata_key(road, direction)
        if check_date_exists(road_key, target_date):
            print(f"{road_key} already has data for {target_date}")
            results[road_key] = True  # Mark as success since data exists
        else:
            print(f"{road_key} missing data for {target_date}, will download...")
            roads_to_download.append((road, direction))

    # If no roads need downloading, return early
    if not roads_to_download:
        print("All roads already have data for this date!")
        return results

    driver = None
    try:
        # Create driver and login once, only if needed
        print(f"Creating browser session for {len(roads_to_download)} roads...")
        driver = create_chrome_driver()
        login_to_pems(driver)

        # Download data for roads that need it
        for road, direction in roads_to_download:
            print(f"Processing {road} {direction}...")
            success = download_road_data(driver, road, direction, target_date)
            road_key = convert_to_metadata_key(road, direction)
            results[road_key] = success

            # Small delay between downloads
            time.sleep(10)

        return results

    except Exception as e:
        print(f"Error in collect_roads_data: {e}")
        return results

    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass

# Example usage
if __name__ == "__main__":
    # Example: Download data for multiple roads
    roads_to_download = [
        ('405', 'N'),
        ('405', 'S'), 
        ('101', 'N'),
        ('134', 'E')
    ]
    
    results = collect_roads_data(roads_to_download, '2025-04-15')
    
    print("\nDownload Results:")
    for road_dir, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"{road_dir}: {status}")