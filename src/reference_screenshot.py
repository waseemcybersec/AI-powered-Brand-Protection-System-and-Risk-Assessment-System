import os
import sys
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# Try to use webdriver-manager if available, otherwise use system chromedriver
try:
    from webdriver_manager.chrome import ChromeDriverManager
    USE_WEBDRIVER_MANAGER = True
except ImportError:
    USE_WEBDRIVER_MANAGER = False

def take_screenshot(url, save_path, driver):
    try:
        driver.get(url)
        time.sleep(3)
        
        # Check if we're on an error page and try to bypass it
        page_source = driver.page_source.lower()
        error_indicators = [
            "your connection is not private",
            "this site can't be reached",
            "err_cert",
            "net::err",
            "privacy error",
            "security warning"
        ]
        
        is_error_page = any(indicator in page_source for indicator in error_indicators)
        
        if is_error_page:
            try:
                # Try to find and click proceed button
                from selenium.webdriver.support.ui import WebDriverWait
                from selenium.webdriver.support import expected_conditions as EC
                
                proceed_selectors = [
                    "//button[contains(text(), 'Advanced')]",
                    "//button[contains(text(), 'Proceed')]",
                    "//a[contains(text(), 'Advanced')]",
                    "//a[contains(text(), 'Proceed')]",
                    "button[id*='proceed']",
                    "a[id*='proceed']",
                ]
                
                for selector in proceed_selectors:
                    try:
                        if selector.startswith("//"):
                            element = driver.find_element(By.XPATH, selector)
                        else:
                            element = driver.find_element(By.CSS_SELECTOR, selector)
                        driver.execute_script("arguments[0].scrollIntoView(true);", element)
                        time.sleep(0.5)
                        element.click()
                        time.sleep(2)
                        break
                    except:
                        continue
            except Exception as e:
                print(f"[WARNING] Could not bypass error page: {e}")
        
        driver.save_screenshot(save_path)
        print(f"[+] Saved screenshot: {save_path}")
    except Exception as e:
        print(f"[!] Failed to capture {url}: {e}")

def main():
    # ==== USER ENTERS BRAND DOMAIN ONLY ====
    brand_input = input("Enter real brand domain (e.g., amazon.com): ").strip()

    if not brand_input.startswith("http"):
        base_url = "https://" + brand_input
    else:
        base_url = brand_input

    # ==== OUTPUT FOLDER ====
    # Get script directory and create proper path
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    output_folder = project_root / "data" / "reference"
    output_folder.mkdir(parents=True, exist_ok=True)

    # ==== SELENIUM SETUP ====
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Bypass SSL certificate errors and security warnings
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_argument("--ignore-ssl-errors")
    chrome_options.add_argument("--ignore-certificate-errors-spki-list")
    chrome_options.add_argument("--allow-insecure-localhost")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--allow-running-insecure-content")
    chrome_options.add_argument("--test-type")
    
    # Set Chrome preferences
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    prefs = {
        "profile.default_content_setting_values": {
            "notifications": 2,
        }
    }
    chrome_options.add_experimental_option("prefs", prefs)
    
    # Windows-compatible ChromeDriver setup
    if USE_WEBDRIVER_MANAGER:
        # Automatically download and manage ChromeDriver
        service = Service(ChromeDriverManager().install())
    else:
        # Try to find chromedriver in common locations or use system PATH
        service = None
        chromedriver_paths = [
            "chromedriver.exe",  # Windows executable in PATH
            "chromedriver",      # Linux/Mac executable in PATH
            os.path.join(os.path.expanduser("~"), "chromedriver"),
            os.path.join(os.path.expanduser("~"), "chromedriver.exe"),
        ]
        for path in chromedriver_paths:
            if os.path.exists(path):
                service = Service(path)
                break
        
        if service is None:
            # Last resort: let Selenium find it in PATH
            service = Service()
    
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # ==== TAKE HOMEPAGE SCREENSHOT ====
    print("\n[+] Capturing Homepage…")
    take_screenshot(base_url, str(output_folder / "homepage.png"), driver)

    driver.quit()
    print("\n[✓] Reference screenshot complete.")

if __name__ == "__main__":
    main()

