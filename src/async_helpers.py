"""
Async helpers for parallel processing of HTTP info and screenshots
Uses multi-processing + multi-threading + async for maximum speed
"""

import asyncio
import concurrent.futures
import multiprocessing
from functools import partial
from pathlib import Path

# Import the original functions
from http_info import get_http_info, is_real_brand
from screenshot import capture_screenshot


# Module-level functions for multiprocessing (must be at module level to be picklable)
def _process_domain_http(domain_brand_tuple):
    """Process a single domain for HTTP info (must be at module level for multiprocessing)"""
    domain, brand_domain = domain_brand_tuple
    try:
        http_data = get_http_info(domain)
        http_data['domain'] = domain
        http_data['real_brand'] = is_real_brand(
            domain,
            http_data.get('final_url', ''),
            http_data.get('title', ''),
            brand_domain
        )
        return http_data
    except Exception as e:
        print(f"Error analyzing {domain}: {e}")
        return {
            'domain': domain,
            'ip': '',
            'http_status': '',
            'final_url': '',
            'title': '',
            'ssl_issuer': '',
            'ssl_notBefore': '',
            'ssl_notAfter': '',
            'real_brand': 'no'
        }


def _process_screenshot(domain_output_tuple):
    """Process a single screenshot (must be at module level for multiprocessing)"""
    domain, output_folder_str, headless = domain_output_tuple
    try:
        # Convert string back to Path for the function
        output_folder = Path(output_folder_str)
        capture_screenshot(domain, output_folder, headless=headless)
        return True
    except Exception as e:
        print(f"Error capturing screenshot for {domain}: {e}")
        return False


def analyze_http_info_parallel(domains, brand_domain, max_workers=None):
    """
    Analyze HTTP info for multiple domains in parallel using multi-processing.
    
    Args:
        domains: List of domains to analyze
        brand_domain: Brand domain for legitimacy detection
        max_workers: Maximum number of worker processes (default: CPU count)
    
    Returns:
        List of HTTP info dictionaries
    """
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(domains))
    
    # Create tuples of (domain, brand_domain) for the worker function
    domain_tuples = [(domain, brand_domain) for domain in domains]
    
    # Use ProcessPoolExecutor for CPU-bound and I/O-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_process_domain_http, domain_tuples))
    
    return results


def capture_screenshots_parallel(domains, output_folder, max_workers=None, headless=True):
    """
    Capture screenshots for multiple domains in parallel using multi-processing.
    
    Args:
        domains: List of domains to screenshot
        output_folder: Output folder for screenshots
        max_workers: Maximum number of worker processes (default: min(4, CPU count))
        headless: Run browser in headless mode
    
    Returns:
        Number of successful screenshots
    """
    if max_workers is None:
        # Limit to 4 to avoid too many Chrome instances
        max_workers = min(4, multiprocessing.cpu_count())
    
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Create tuples of (domain, output_folder_str, headless) for the worker function
    # Convert Path to string for pickling
    output_folder_str = str(output_folder)
    screenshot_tuples = [(domain, output_folder_str, headless) for domain in domains]
    
    # Use ProcessPoolExecutor - each process runs its own Chrome instance
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_process_screenshot, screenshot_tuples))
    
    return sum(results)

