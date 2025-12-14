"""
Capture multiple reference screenshots from the brand website
Captures: homepage, login page, and other important pages
"""

import os
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Try to use webdriver-manager if available
try:
    from webdriver_manager.chrome import ChromeDriverManager
    USE_WEBDRIVER_MANAGER = True
except ImportError:
    USE_WEBDRIVER_MANAGER = False


def take_screenshot(url, save_path, driver):
    """Take a screenshot of a URL"""
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
        return True
    except Exception as e:
        print(f"[!] Failed to capture {url}: {e}")
        return False


def find_login_page(driver, base_url):
    """Try to find and navigate to login page"""
    login_keywords = ["login", "sign in", "signin", "log in", "account"]
    
    try:
        # Try common login URLs
        login_urls = [
            f"{base_url}/login",
            f"{base_url}/signin",
            f"{base_url}/account/login",
            f"{base_url}/sign-in",
        ]
        
        for login_url in login_urls:
            try:
                driver.get(login_url)
                time.sleep(2)
                page_source = driver.page_source.lower()
                # Check if we're on a login page
                if any(keyword in page_source for keyword in login_keywords):
                    return login_url
            except:
                continue
        
        # Try to find login link on homepage
        try:
            driver.get(base_url)
            time.sleep(2)
            elements = driver.find_elements(By.TAG_NAME, "a")
            
            for el in elements:
                text = el.text.lower().strip()
                href = el.get_attribute("href") or ""
                if any(word in text for word in login_keywords) or any(word in href.lower() for word in login_keywords):
                    login_url = el.get_attribute("href")
                    if login_url and login_url.startswith("http"):
                        return login_url
        except:
            pass
        
        return None
    except Exception:
        return None


def capture_multiple_reference_screenshots(brand_domain, output_folder):
    """
    Capture multiple reference screenshots from the brand website.
    
    Args:
        brand_domain: Brand domain (e.g., instagram.com)
        output_folder: Path to output folder
    
    Returns:
        List of captured screenshot paths
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    if not brand_domain.startswith("http"):
        base_url = "https://" + brand_domain
    else:
        base_url = brand_domain
    
    # Setup Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_argument("--ignore-ssl-errors")
    chrome_options.add_argument("--ignore-certificate-errors-spki-list")
    chrome_options.add_argument("--allow-insecure-localhost")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--allow-running-insecure-content")
    chrome_options.add_argument("--test-type")
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    try:
        if USE_WEBDRIVER_MANAGER:
            service = Service(ChromeDriverManager().install())
        else:
            service = Service()
    except:
        service = Service()
    
    driver = webdriver.Chrome(service=service, options=chrome_options)
    captured_screenshots = []
    
    try:
        # 1. Homepage
        print("[+] Capturing homepage...")
        homepage_path = output_folder / "homepage.png"
        if take_screenshot(base_url, str(homepage_path), driver):
            captured_screenshots.append(homepage_path)
        
        # 2. Login page
        print("[+] Looking for login page...")
        login_url = find_login_page(driver, base_url)
        if login_url:
            print(f"[+] Found login page: {login_url}")
            login_path = output_folder / "login.png"
            if take_screenshot(login_url, str(login_path), driver):
                captured_screenshots.append(login_path)
        else:
            print("[!] Could not find login page")
        
        # 3. Try other common pages
        common_pages = [
            ("/signup", "signup.png"),
            ("/register", "register.png"),
            ("/account", "account.png"),
        ]
        
        for path, filename in common_pages:
            try:
                url = base_url + path
                print(f"[+] Trying {url}...")
                driver.get(url)
                time.sleep(2)
                # Check if page exists (not 404)
                if "404" not in driver.title.lower() and "not found" not in driver.page_source.lower():
                    page_path = output_folder / filename
                    if take_screenshot(url, str(page_path), driver):
                        captured_screenshots.append(page_path)
            except:
                continue
        
    finally:
        driver.quit()
    
    print(f"[✓] Captured {len(captured_screenshots)} reference screenshots")
    return captured_screenshots

