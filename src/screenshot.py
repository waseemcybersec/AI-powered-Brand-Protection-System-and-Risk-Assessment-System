# screenshot.py
#!/usr/bin/env python3
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import argparse
import os
from pathlib import Path

# Try to use webdriver-manager if available
try:
    from webdriver_manager.chrome import ChromeDriverManager
    USE_WEBDRIVER_MANAGER = True
except ImportError:
    USE_WEBDRIVER_MANAGER = False

def capture_screenshot(domain, output_folder=None, headless=True):
    # Get script directory and create proper path
    if output_folder is None:
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir.parent
        output_folder = project_root / "data" / "screenshots"
    else:
        output_folder = Path(output_folder)
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    
    # Bypass SSL certificate errors and security warnings
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--ignore-ssl-errors")
    options.add_argument("--ignore-certificate-errors-spki-list")
    options.add_argument("--allow-insecure-localhost")
    options.add_argument("--disable-web-security")
    options.add_argument("--allow-running-insecure-content")
    options.add_argument("--disable-features=VizDisplayCompositor")
    
    # Additional flags to bypass security warnings
    options.add_argument("--ignore-certificate-errors-spki-list")
    options.add_argument("--ignore-ssl-errors")
    options.add_argument("--test-type")
    options.add_argument("--ignore-certificate-errors")
    
    # Set Chrome preferences to ignore SSL errors
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_experimental_option('useAutomationExtension', False)
    prefs = {
        "profile.default_content_setting_values": {
            "notifications": 2,  # Disable notifications
        },
        "profile.managed_default_content_settings": {
            "images": 1  # Allow images
        }
    }
    options.add_experimental_option("prefs", prefs)
    
    # Windows-compatible ChromeDriver setup
    if USE_WEBDRIVER_MANAGER:
        service = Service(ChromeDriverManager().install())
    else:
        service = None
        chromedriver_paths = [
            "chromedriver.exe",
            "chromedriver",
            os.path.join(os.path.expanduser("~"), "chromedriver"),
            os.path.join(os.path.expanduser("~"), "chromedriver.exe"),
        ]
        for path in chromedriver_paths:
            if os.path.exists(path):
                service = Service(path)
                break
        if service is None:
            service = Service()
    
    driver = webdriver.Chrome(service=service, options=options)
    try:
        # Try HTTPS first, fallback to HTTP
        loaded = False
        for protocol in ["https", "http"]:
            try:
                url = f"{protocol}://{domain}"
                driver.get(url)
                
                # Wait for page to load
                import time
                time.sleep(3)  # Give page time to load
                
                # Check if we're on an error page
                page_source = driver.page_source.lower()
                current_url = driver.current_url.lower()
                
                # Common error page indicators
                error_indicators = [
                    "your connection is not private",
                    "this site can't be reached",
                    "err_cert",
                    "net::err",
                    "privacy error",
                    "security warning"
                ]
                
                is_error_page = any(indicator in page_source for indicator in error_indicators)
                
                # If it's an error page and we're trying HTTPS, try to proceed anyway
                if is_error_page and protocol == "https":
                    try:
                        # Try to find and click "Advanced" or "Proceed" button
                        from selenium.webdriver.common.by import By
                        from selenium.webdriver.support.ui import WebDriverWait
                        from selenium.webdriver.support import expected_conditions as EC
                        
                        # Look for common proceed buttons
                        proceed_selectors = [
                            "button[id*='proceed']",
                            "button[id*='advanced']",
                            "a[id*='proceed']",
                            "a[id*='advanced']",
                            "//button[contains(text(), 'Advanced')]",
                            "//button[contains(text(), 'Proceed')]",
                            "//a[contains(text(), 'Advanced')]",
                            "//a[contains(text(), 'Proceed')]",
                            "//button[contains(text(), 'Continue')]",
                            "//span[contains(text(), 'Advanced')]",
                            "//span[contains(text(), 'Proceed')]",
                        ]
                        
                        clicked = False
                        for selector in proceed_selectors:
                            try:
                                if selector.startswith("//"):
                                    element = driver.find_element(By.XPATH, selector)
                                else:
                                    element = driver.find_element(By.CSS_SELECTOR, selector)
                                # Scroll to element and click
                                driver.execute_script("arguments[0].scrollIntoView(true);", element)
                                time.sleep(0.5)
                                element.click()
                                time.sleep(2)  # Wait after clicking
                                clicked = True
                                break
                            except:
                                continue
                        
                        # If clicking didn't work, try JavaScript navigation
                        if not clicked:
                            try:
                                # Try to find the proceed link via JavaScript
                                proceed_link = driver.execute_script("""
                                    var links = document.querySelectorAll('a, button');
                                    for (var i = 0; i < links.length; i++) {
                                        var text = links[i].textContent || links[i].innerText || '';
                                        var id = links[i].id || '';
                                        if (text.toLowerCase().includes('proceed') || 
                                            text.toLowerCase().includes('advanced') ||
                                            id.toLowerCase().includes('proceed') ||
                                            id.toLowerCase().includes('advanced')) {
                                            return links[i];
                                        }
                                    }
                                    return null;
                                """)
                                if proceed_link:
                                    driver.execute_script("arguments[0].click();", proceed_link)
                                    time.sleep(2)
                            except:
                                pass
                        
                        # Re-check if still error page
                        time.sleep(1)
                        page_source = driver.page_source.lower()
                        is_error_page = any(indicator in page_source for indicator in error_indicators)
                        
                        # Last resort: try to navigate directly using JavaScript
                        if is_error_page:
                            try:
                                # Try to navigate directly, bypassing the error page
                                driver.execute_script(f"window.location.href = '{url}';")
                                time.sleep(3)
                                page_source = driver.page_source.lower()
                                is_error_page = any(indicator in page_source for indicator in error_indicators)
                            except:
                                pass
                        
                    except Exception as e:
                        print(f"[WARNING] Could not bypass error page for {domain}: {e}")
                
                # If still error page, try HTTP instead
                if is_error_page and protocol == "https":
                    continue
                
                loaded = True
                break
                
            except Exception as e:
                print(f"[DEBUG] Failed {protocol}://{domain}: {e}")
                continue
        
        if not loaded:
            raise Exception(f"Could not connect to {domain} via HTTP or HTTPS")
        
        # Additional wait to ensure page is fully loaded
        import time
        time.sleep(2)
        
        path = output_folder / f"{domain.replace('.','_')}.png"
        driver.save_screenshot(str(path))
        print(f"[INFO] Screenshot saved: {path}")
    except Exception as e:
        print(f"[ERROR] {domain} -> {e}")
    finally:
        driver.quit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="TXT file with domains")
    parser.add_argument("--output_folder", default=None, help="Output folder for screenshots (default: ../data/screenshots)")
    parser.add_argument("--no-headless", action="store_true", help="Run browser in visible mode (useful for debugging)")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.is_absolute():
        input_path = Path(__file__).parent / args.input_file
    
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return
    
    # Determine output folder
    if args.output_folder is None:
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir.parent
        output_folder = project_root / "data" / "screenshots"
    else:
        output_folder = Path(args.output_folder)
    
    # Create folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Delete all existing PNG files from the folder
    print(f"[INFO] Cleaning up existing screenshots in {output_folder}...")
    deleted_count = 0
    for png_file in output_folder.glob("*.png"):
        try:
            png_file.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"[WARNING] Could not delete {png_file.name}: {e}")
    
    if deleted_count > 0:
        print(f"[INFO] Deleted {deleted_count} existing screenshot(s)")
    else:
        print(f"[INFO] No existing screenshots found")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        domains = [line.strip() for line in f if line.strip()]
    
    print(f"\n[INFO] Processing {len(domains)} domains...")
    headless_mode = not args.no_headless
    for i, domain in enumerate(domains, 1):
        print(f"[{i}/{len(domains)}] Processing {domain}...")
        capture_screenshot(domain, args.output_folder, headless=headless_mode)

if __name__ == "__main__":
    main()

