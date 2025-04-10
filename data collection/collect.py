import time
import os
from urllib.parse import urlparse, parse_qs
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# Specify the download directory
download_path = '/Users/amitomer/Downloads/data collection/data'

# Set up Chrome options
chrome_options = Options()
chrome_prefs = {
    "download.default_directory": download_path,  # Change the download directory
    "download.prompt_for_download": False,  # Disable download prompt
    "download.directory_upgrade": True,  # Automatically upgrade download path
}
chrome_options.add_experimental_option("prefs", chrome_prefs)

# Set up ChromeDriver with the path to your chromedriver and options
driver = webdriver.Chrome(service=Service('/opt/homebrew/bin/chromedriver'), options=chrome_options)

# Continue with your script as normal...

# Open the login page
url = "https://pems.dot.ca.gov/"
driver.get(url)

# Wait for the page to load
time.sleep(20)

# Find the username and password fields
username_field = driver.find_element(By.ID, "username")
password_field = driver.find_element(By.ID, "password")

# Enter your credentials
username_field.send_keys("amitomer1912@gmail.com")
password_field.send_keys("5^applel?X")

# Find and click the login button (using the 'login' name attribute)
login_button = driver.find_element(By.NAME, "login")
login_button.click()

# Wait for login to complete
time.sleep(20)
i = 2
s_time_id=1740873600
while(i <= 31):
    # Once logged in, navigate to the report page
    if i <= 9:
        day = f"0{i}"
    else:
        day = i
    report_url = f"https://pems.dot.ca.gov/?report_form=1&dnode=Freeway&content=spatial&tab=contours&export=&fwy=405&dir=N&s_time_id={s_time_id}&s_time_id_f=03%2F{day}%2F2025&from_hh=0&to_hh=23&start_pm=.36&end_pm=72.09&lanes=&station_type=ml&q=speed&colormap=30%2C31%2C32&sc=auto&ymin=&ymax=&view_d=2&chart.x=93&chart.y=20"
    driver.get(report_url)
    time.sleep(60)

    # Find the "Export XLS" button by its 'name' or 'alt' attribute
    export_button = driver.find_element(By.NAME, "xls")

    # Click the "Export XLS" button
    export_button.click()

    # Wait for the download to complete (you can increase or decrease this based on your network speed)
    time.sleep(100)

    # Extract highway number and date from the report URL
    parsed_url = urlparse(report_url)
    query_params = parse_qs(parsed_url.query)
    highway_number = query_params.get('fwy', ['unknown'])[0]  # Default to 'unknown' if not found
    date_taken = query_params.get('s_time_id_f', ['unknown_date'])[0]  # Default to 'unknown_date' if not found

    # Construct the new file name
    new_file_name = rf"{highway_number}_{date_taken.replace("/", "*")}.xlsx"
    new_file_path = os.path.join(download_path, new_file_name)

    # Original file path (before renaming)
    original_file_path = os.path.join(download_path, 'pems_output.xlsx')

    # Rename the file if it exists
    if os.path.exists(original_file_path):
        os.rename(original_file_path, new_file_path)
        print(f"File renamed to: {new_file_path}")
    else:
        print("Original file not found!")

    print(f"Download {i} to March complete!")
    i+=1
    s_time_id += 86400
    
driver.quit()
