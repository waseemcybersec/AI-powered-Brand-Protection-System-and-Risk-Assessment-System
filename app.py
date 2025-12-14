#!/usr/bin/env python3
"""
Brand Protection Web Application
A beautiful web interface for the brand protection pipeline.
"""

import os
import sys
import json
import threading
import time
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import pandas as pd

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install it with: pip install python-dotenv")

# Add src directory to path to import pipeline modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import pipeline modules
try:
    from generate import generate
    from screenshot import capture_screenshot
    from reference_screenshot import take_screenshot
    from mimic_detection import MimicDetector
    from logo_detection import LogoDetector, find_reference_logos, get_brand_name_from_domain
    from threat_intelligence import ThreatIntelligenceCollector
    from dark_web_monitoring import DarkWebMonitor
    from social_media_monitoring import SocialMediaMonitor
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'brand-protection-secret-key-2024'

# Task status storage (in production, use Redis or database)
task_status = {}
task_results = {}
ti_results = {}  # Threat intelligence results
dw_results = {}  # Dark web monitoring results
sm_results = {}  # Social media monitoring results
logo_results = {}  # Logo detection results
mimic_results = {}  # Mimic detection results

# Project paths
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"
REFERENCE_DIR = DATA_DIR / "reference"
REFERENCE_LOGOS_DIR = DATA_DIR / "reference_logos"


def update_task_status(task_id, status, message="", progress=0, data=None):
    """Update task status"""
    task_status[task_id] = {
        'status': status,  # 'running', 'completed', 'error'
        'message': message,
        'progress': progress,
        'timestamp': datetime.now().isoformat(),
        'data': data or {}
    }


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/api/status/<task_id>')
def get_task_status(task_id):
    """Get status of a running task"""
    if task_id in task_status:
        return jsonify(task_status[task_id])
    return jsonify({'status': 'not_found'}), 404


@app.route('/api/pipeline/run', methods=['POST'])
def run_pipeline():
    """Run the complete brand protection pipeline"""
    data = request.json
    brand_domain = data.get('domain', '').strip()
    
    if not brand_domain:
        return jsonify({'error': 'Domain is required'}), 400
    
    # Generate task ID
    task_id = f"pipeline_{int(time.time())}"
    update_task_status(task_id, 'running', 'Starting pipeline...', 0)
    
    # Run pipeline in background thread
    thread = threading.Thread(
        target=run_pipeline_background,
        args=(task_id, brand_domain, data)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'task_id': task_id})


def run_pipeline_background(task_id, brand_domain, options):
    """Run pipeline in background thread"""
    try:
        # Determine detection mode first - EXCLUSIVE: only one mode can run
        detection_mode = options.get('detection_mode', 'mimic')
        print(f"[DEBUG] Detection mode from options: {detection_mode}")
        print(f"[DEBUG] Options: run_mimic_detection={options.get('run_mimic_detection', False)}, run_logo_detection={options.get('run_logo_detection', False)}")
        
        # EXCLUSIVE mode selection - only one can be True
        run_mimic = (detection_mode == 'mimic') and options.get('run_mimic_detection', False)
        run_logo = (detection_mode == 'logo') and options.get('run_logo_detection', False)
        run_threat = (detection_mode == 'threat') and options.get('run_threat_intelligence', False)
        run_darkweb = (detection_mode == 'darkweb') and options.get('run_dark_web_monitoring', False)
        run_socialmedia = (detection_mode == 'socialmedia') and options.get('run_social_media_monitoring', False)
        
        # Safety check: ensure only one mode is selected
        modes_selected = sum([run_mimic, run_logo, run_threat, run_darkweb, run_socialmedia])
        if modes_selected > 1:
            print(f"[WARNING] Multiple detection modes selected! Forcing exclusive mode: {detection_mode}")
            run_mimic = (detection_mode == 'mimic')
            run_logo = (detection_mode == 'logo')
            run_threat = (detection_mode == 'threat')
            run_darkweb = (detection_mode == 'darkweb')
            run_socialmedia = (detection_mode == 'socialmedia')
        
        print(f"[DEBUG] Final flags: run_mimic={run_mimic}, run_logo={run_logo}, run_threat={run_threat}, run_darkweb={run_darkweb}, run_socialmedia={run_socialmedia}")
        
        # Step 7: Detection (Threat Intelligence - skip domain generation)
        if run_threat:
            # Threat Intelligence mode - skip domain generation and detection
            update_task_status(task_id, 'running', 'Collecting threat intelligence...', 10)
            try:
                # Get API keys from environment variables (or from options as fallback)
                api_keys = {
                    'virustotal': os.getenv('VIRUSTOTAL_API_KEY', options.get('api_keys', {}).get('virustotal', '')),
                    'abuseipdb': os.getenv('ABUSEIPDB_API_KEY', options.get('api_keys', {}).get('abuseipdb', '')),
                    'safebrowsing': os.getenv('SAFEBROWSING_API_KEY', options.get('api_keys', {}).get('safebrowsing', ''))
                }
                
                update_task_status(task_id, 'running', 'Querying threat intelligence sources...', 30)
                collector = ThreatIntelligenceCollector(api_keys=api_keys)
                ti_result = collector.collect_all(brand_domain)
                
                update_task_status(task_id, 'running', 'Generating threat intelligence report...', 90)
                # Generate report
                report_path = SRC_DIR / f"threat_intelligence_report_{brand_domain.replace('.', '_')}.html"
                collector.generate_report(ti_result, report_path)
                
                update_task_status(
                    task_id,
                    'completed',
                    f'Threat intelligence analysis complete! Risk Score: {ti_result.get("risk_score", 0)}/100',
                    100,
                    {
                        'domain': brand_domain,
                        'risk_score': ti_result.get('risk_score', 0),
                        'risk_level': ti_result.get('risk_level', 'Unknown'),
                        'report_file': str(report_path.relative_to(PROJECT_ROOT))
                    }
                )
                ti_results[task_id] = ti_result
                return
            except Exception as e:
                import traceback
                error_msg = str(e)
                traceback.print_exc()
                update_task_status(task_id, 'error', f'Threat intelligence failed: {error_msg}', 0)
                return
        
        # Step 7b: Dark Web Monitoring (skip domain generation)
        elif run_darkweb:
            update_task_status(task_id, 'running', 'Monitoring dark web sources...', 10)
            try:
                # Get API keys from environment variables
                api_keys = {
                    'intelx': os.getenv('INTELX_API_KEY', ''),
                    'dehashed': os.getenv('DEHASHED_API_KEY', ''),
                    'dehashed_email': os.getenv('DEHASHED_EMAIL', ''),  # Optional: DeHashed account email
                    'hibp': os.getenv('HIBP_API_KEY', '')  # Optional
                }
                
                update_task_status(task_id, 'running', 'Querying IntelligenceX, DeHashed, and other sources...', 30)
                monitor = DarkWebMonitor(api_keys=api_keys)
                brand_name = options.get('brand_name', None)
                dw_result = monitor.monitor_brand(brand_domain, brand_name)
                
                update_task_status(task_id, 'running', 'Generating dark web monitoring report...', 90)
                # Generate report
                report_path = SRC_DIR / f"dark_web_report_{brand_domain.replace('.', '_')}.html"
                monitor.generate_report(dw_result, report_path)
                
                update_task_status(
                    task_id,
                    'completed',
                    f'Dark web monitoring complete! Risk Score: {dw_result.get("risk_score", 0)}/100',
                    100,
                    {
                        'domain': brand_domain,
                        'brand_name': dw_result.get('brand_name', brand_domain),
                        'risk_score': dw_result.get('risk_score', 0),
                        'risk_level': dw_result.get('risk_level', 'Unknown'),
                        'report_file': str(report_path.relative_to(PROJECT_ROOT))
                    }
                )
                dw_results[task_id] = dw_result
                return
            except Exception as e:
                import traceback
                error_msg = str(e)
                traceback.print_exc()
                update_task_status(task_id, 'error', f'Dark web monitoring failed: {error_msg}', 0)
                return
        
        # Step 7c: Social Media Monitoring (skip domain generation)
        elif run_socialmedia:
            update_task_status(task_id, 'running', 'Monitoring social media platforms...', 10)
            try:
                # Get API keys from environment variables
                api_keys = {
                    'youtube': os.getenv('YOUTUBE_API_KEY', ''),
                    'telegram': os.getenv('TELEGRAM_BOT_TOKEN', ''),
                    'github': os.getenv('GITHUB_TOKEN', '')
                }
                
                update_task_status(task_id, 'running', 'Checking YouTube, Telegram, and GitHub...', 30)
                monitor = SocialMediaMonitor(api_keys=api_keys)
                brand_name = options.get('brand_name', None)
                sm_result = monitor.monitor_brand(brand_domain, brand_name)
                
                update_task_status(task_id, 'running', 'Generating social media monitoring report...', 90)
                # Generate report
                report_path = SRC_DIR / f"social_media_report_{brand_domain.replace('.', '_')}.html"
                monitor.generate_report(sm_result, report_path)
                
                update_task_status(
                    task_id,
                    'completed',
                    f'Social media monitoring complete! Risk Score: {sm_result.get("risk_score", 0)}/100',
                    100,
                    {
                        'domain': brand_domain,
                        'brand_name': sm_result.get('brand_name', brand_domain),
                        'risk_score': sm_result.get('risk_score', 0),
                        'risk_level': sm_result.get('risk_level', 'Unknown'),
                        'report_file': str(report_path.relative_to(PROJECT_ROOT))
                    }
                )
                sm_results[task_id] = sm_result
                return
            except Exception as e:
                import traceback
                error_msg = str(e)
                traceback.print_exc()
                update_task_status(task_id, 'error', f'Social media monitoring failed: {error_msg}', 0)
                return
        
        # Step 1: Generate domain candidates (for mimic/logo detection)
        update_task_status(task_id, 'running', 'Generating domain candidates...', 10)
        candidates_file = SRC_DIR / "candidates.txt"
        variants = generate(brand_domain, tlds=options.get('tlds', ['com', 'net', 'org', 'co', 'info', 'io', 'xyz', 'shop']))
        
        with open(candidates_file, 'w', encoding='utf-8') as f:
            for variant in variants:
                f.write(f"{variant}\n")
        
        update_task_status(task_id, 'running', f'Generated {len(variants)} candidates', 20)
        
        # Clean screenshot folders if detection is enabled
        if run_mimic or run_logo:
            update_task_status(task_id, 'running', 'Cleaning screenshot folders...', 21)
            # Clean screenshots folder
            if SCREENSHOTS_DIR.exists():
                for png_file in SCREENSHOTS_DIR.glob("*.png"):
                    try:
                        png_file.unlink()
                    except Exception as e:
                        print(f"Warning: Could not delete {png_file}: {e}")
            
            # Clean reference folder (only for mimic detection - NOT for logo detection)
            if run_mimic and not run_logo and REFERENCE_DIR.exists():
                for png_file in REFERENCE_DIR.glob("*.png"):
                    try:
                        png_file.unlink()
                    except Exception as e:
                        print(f"Warning: Could not delete {png_file}: {e}")
        
        # Step 2: DNS Check
        update_task_status(task_id, 'running', 'Checking DNS for live domains...', 30)
        
        def dns_progress(count):
            if count % 10 == 0:  # Update every 10 domains
                update_task_status(task_id, 'running', f'Checking DNS... Found {count} live domains so far', 30 + int(count / len(variants) * 20))
        
        live_domains = check_dns_live(candidates_file, progress_callback=dns_progress)
        live_file = SRC_DIR / "live_domains.txt"
        with open(live_file, 'w', encoding='utf-8') as f:
            for domain in live_domains:
                f.write(f"{domain}\n")
        
        update_task_status(task_id, 'running', f'Found {len(live_domains)} live domains', 50)
        
        # Step 3: HTTP Analysis (parallel)
        update_task_status(task_id, 'running', 'Analyzing HTTP information (parallel)...', 60)
        from async_helpers import analyze_http_info_parallel
        http_features = analyze_http_info_parallel(live_domains, brand_domain)
        http_file = SRC_DIR / "http_features.csv"
        pd.DataFrame(http_features).to_csv(http_file, index=False, encoding='utf-8')
        update_task_status(task_id, 'running', 'HTTP analysis complete', 70)
        
        # Step 4: Capture Screenshots (parallel, always if detection is enabled)
        if run_mimic or run_logo:
            update_task_status(task_id, 'running', 'Capturing screenshots (parallel)...', 75)
            from async_helpers import capture_screenshots_parallel
            screenshot_count = capture_screenshots_parallel(live_domains, SCREENSHOTS_DIR, headless=True)
            update_task_status(task_id, 'running', f'Captured {screenshot_count}/{len(live_domains)} screenshots', 85)
        
        # Step 5: Multiple Reference Screenshots (ONLY for mimic detection - NEVER for logo detection)
        reference_image_paths = []
        if run_mimic and not run_logo:
            update_task_status(task_id, 'running', 'Capturing multiple reference screenshots...', 87)
            REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
            try:
                from reference_screenshots_multi import capture_multiple_reference_screenshots
                captured_refs = capture_multiple_reference_screenshots(brand_domain, REFERENCE_DIR)
                reference_image_paths = [str(p) for p in captured_refs]
                update_task_status(task_id, 'running', f'Captured {len(reference_image_paths)} reference screenshots', 90)
            except Exception as e:
                print(f"Error capturing reference screenshots: {e}")
                # Fallback to single homepage screenshot
                try:
                    from reference_screenshot import take_screenshot
                    from selenium import webdriver
                    from selenium.webdriver.chrome.service import Service
                    from selenium.webdriver.chrome.options import Options
                    try:
                        from webdriver_manager.chrome import ChromeDriverManager
                        service = Service(ChromeDriverManager().install())
                    except:
                        service = Service()
                    
                    chrome_options = Options()
                    chrome_options.add_argument("--headless=new")
                    chrome_options.add_argument("--no-sandbox")
                    chrome_options.add_argument("--disable-dev-shm-usage")
                    chrome_options.add_argument("--window-size=1920,1080")
                    chrome_options.add_argument("--ignore-certificate-errors")
                    chrome_options.add_argument("--ignore-ssl-errors")
                    
                    driver = webdriver.Chrome(service=service, options=chrome_options)
                    base_url = f"https://{brand_domain}" if not brand_domain.startswith("http") else brand_domain
                    homepage_path = REFERENCE_DIR / "homepage.png"
                    take_screenshot(base_url, str(homepage_path), driver)
                    driver.quit()
                    reference_image_paths = [str(homepage_path)]
                    update_task_status(task_id, 'running', 'Captured homepage reference screenshot', 90)
                except Exception as e2:
                    print(f"Fallback also failed: {e2}")
                    update_task_status(task_id, 'running', f'Warning: Could not capture reference screenshots: {str(e)}', 90)
        
        # Step 6: Compile Final Dataset
        update_task_status(task_id, 'running', 'Compiling final dataset...', 90)
        dataset_file = compile_final_dataset(http_file)
        
        # Step 7: Detection (Mimic or Logo - run ONLY one, not both)
        # CRITICAL: Check detection_mode directly to ensure exclusivity - these are MUTUALLY EXCLUSIVE
        print(f"[DEBUG] Step 7: detection_mode={detection_mode}, run_mimic={run_mimic}, run_logo={run_logo}, reference_image_paths={len(reference_image_paths) if reference_image_paths else 0}")
        
        # MIMIC DETECTION: Only run if detection_mode is explicitly 'mimic'
        if detection_mode == 'mimic':
            if not run_mimic:
                print(f"[ERROR] Mimic detection mode selected but run_mimic_detection is False!")
                update_task_status(task_id, 'error', 'Mimic detection mode selected but not enabled', 0)
                return
            
            if not reference_image_paths:
                print(f"[ERROR] Mimic detection requires reference images but none were captured!")
                update_task_status(task_id, 'error', 'Mimic detection failed: No reference images captured', 0)
                return
            
            print(f"[INFO] Running MIMIC detection (mode: {detection_mode})")
            update_task_status(task_id, 'running', 'Running mimic detection (comparing against all reference images)...', 95)
            try:
                detector = MimicDetector(
                    reference_image_paths,  # Pass all reference images
                    phash_threshold=options.get('phash_threshold', 10),
                    clip_threshold=options.get('clip_threshold', 0.80)  # Default: 0.80 (80%)
                )
                df = pd.read_csv(dataset_file, encoding='utf-8')
                df_result = detector.process_dataset(df, screenshot_base_path=PROJECT_ROOT)
                output_file = SRC_DIR / "dataset_with_mimic.csv"
                df_result.to_csv(output_file, index=False, encoding='utf-8')
                dataset_file = output_file
                
                # Store results for report generation
                results_dict = df_result.to_dict('records')
                mimic_results[task_id] = {
                    'brand_domain': brand_domain,
                    'results': results_dict,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Generate HTML report
                report_path = SRC_DIR / f"mimic_detection_report_{brand_domain.replace('.', '_')}.html"
                from mimic_detection import MimicDetector
                MimicDetector.generate_report(results_dict, brand_domain, report_path)
                print(f"[INFO] Mimic detection completed successfully")
            except Exception as e:
                print(f"[ERROR] Error in mimic detection: {e}")
                import traceback
                traceback.print_exc()
                update_task_status(task_id, 'error', f'Mimic detection failed: {str(e)}', 0)
                return
        
        # LOGO DETECTION: Only run if detection_mode is explicitly 'logo'
        elif detection_mode == 'logo':
            if not run_logo:
                print(f"[ERROR] Logo detection mode selected but run_logo_detection is False!")
                update_task_status(task_id, 'error', 'Logo detection mode selected but not enabled', 0)
                return
            
            print(f"[INFO] Running LOGO detection (mode: {detection_mode})")
            update_task_status(task_id, 'running', 'Running logo detection...', 92)
            try:
                # Check for reference logo folder (automatically finds logos in data/reference_logos/<brandname>/)
                try:
                    reference_logo_paths = find_reference_logos(brand_domain, PROJECT_ROOT)
                    update_task_status(task_id, 'running', f'Found {len(reference_logo_paths)} reference logo(s)', 93)
                except FileNotFoundError as e:
                    error_msg = str(e)
                    update_task_status(task_id, 'error', f'Logo detection failed: {error_msg}', 0)
                    return
                
                # Initialize logo detector with EXTREMELY STRICT ensemble approach
                # Requires ALL 3 methods to agree on same reference with high confidence
                detector = LogoDetector(
                    reference_logo_paths,
                    similarity_threshold=options.get('similarity_threshold', 0.88),  # EXTREMELY STRICT: 88% DINOv2 threshold
                    phash_threshold=options.get('phash_threshold', 2),  # EXTREMELY STRICT: pHash distance <= 2
                    template_threshold=options.get('template_threshold', 0.88)  # EXTREMELY STRICT: 88% template matching
                )
                
                df = pd.read_csv(dataset_file, encoding='utf-8')
                df_result = detector.process_dataset(df, screenshot_base_path=PROJECT_ROOT)
                output_file = SRC_DIR / "dataset_with_logo.csv"
                df_result.to_csv(output_file, index=False, encoding='utf-8')
                dataset_file = output_file
                
                # Store results for report generation
                results_dict = df_result.to_dict('records')
                logo_results[task_id] = {
                    'brand_domain': brand_domain,
                    'results': results_dict,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Generate HTML report
                report_path = SRC_DIR / f"logo_detection_report_{brand_domain.replace('.', '_')}.html"
                from logo_detection import LogoDetector
                LogoDetector.generate_report(results_dict, brand_domain, report_path)
                
                update_task_status(task_id, 'running', 'Logo detection complete', 98)
            except Exception as e:
                print(f"Error in logo detection: {e}")
                import traceback
                traceback.print_exc()
                update_task_status(task_id, 'error', f'Logo detection failed: {str(e)}', 0)
                return
        
        # CRITICAL: If logo detection was selected, ensure mimic detection did NOT run
        if detection_mode == 'logo' and run_logo:
            print(f"[DEBUG] Logo detection completed. Ensuring we use logo detection results only.")
            # Force use of logo detection file
            dataset_file = SRC_DIR / "dataset_with_logo.csv"
            if not dataset_file.exists():
                print(f"[ERROR] Logo detection file not found: {dataset_file}")
                update_task_status(task_id, 'error', 'Logo detection file not found', 0)
                return
        
        # CRITICAL: If mimic detection was selected, ensure logo detection did NOT run
        if detection_mode == 'mimic' and run_mimic:
            print(f"[DEBUG] Mimic detection completed. Ensuring we use mimic detection results only.")
            # Force use of mimic detection file
            dataset_file = SRC_DIR / "dataset_with_mimic.csv"
            if not dataset_file.exists():
                print(f"[ERROR] Mimic detection file not found: {dataset_file}")
                update_task_status(task_id, 'error', 'Mimic detection file not found', 0)
                return
        
        # Load final results - CRITICAL: ensure we load the correct file based on detection mode
        print(f"[DEBUG] ===== LOADING FINAL RESULTS =====")
        print(f"[DEBUG] Detection mode: {detection_mode}")
        print(f"[DEBUG] Current dataset_file: {dataset_file}")
        
        # CRITICAL: Force correct file based on detection mode - OVERRIDE any previous assignment
        if detection_mode == 'logo':
            # Logo detection mode - MUST use logo detection file, IGNORE anything else
            logo_file = SRC_DIR / "dataset_with_logo.csv"
            if logo_file.exists():
                print(f"[INFO] ✓ Logo detection mode: FORCING use of {logo_file}")
                dataset_file = logo_file
            else:
                print(f"[ERROR] Logo detection file not found: {logo_file}")
                update_task_status(task_id, 'error', f'Logo detection file not found: {logo_file}', 0)
                return
        elif detection_mode == 'mimic':
            # Mimic detection mode - MUST use mimic detection file, IGNORE anything else
            mimic_file = SRC_DIR / "dataset_with_mimic.csv"
            if mimic_file.exists():
                print(f"[INFO] ✓ Mimic detection mode: FORCING use of {mimic_file}")
                dataset_file = mimic_file
            else:
                print(f"[ERROR] Mimic detection file not found: {mimic_file}")
                update_task_status(task_id, 'error', f'Mimic detection file not found: {mimic_file}', 0)
                return
        
        print(f"[DEBUG] Final dataset_file to load: {dataset_file}")
        df_final = pd.read_csv(dataset_file, encoding='utf-8')
        
        # Verify the loaded file has the correct columns
        if detection_mode == 'logo':
            if 'logo_detected' not in df_final.columns:
                print(f"[ERROR] Logo detection file loaded but 'logo_detected' column not found!")
                print(f"[ERROR] Columns found: {list(df_final.columns)}")
                update_task_status(task_id, 'error', 'Loaded file does not contain logo detection results', 0)
                return
            print(f"[INFO] ✓ Verified: File contains logo_detected column")
        elif detection_mode == 'mimic':
            if 'mimic_brand' not in df_final.columns:
                print(f"[ERROR] Mimic detection file loaded but 'mimic_brand' column not found!")
                print(f"[ERROR] Columns found: {list(df_final.columns)}")
                update_task_status(task_id, 'error', 'Loaded file does not contain mimic detection results', 0)
                return
            print(f"[INFO] ✓ Verified: File contains mimic_brand column")
        # Replace NaN/NaT with None for JSON serialization (avoid fillna to prevent version issues)
        import numpy as np
        import math
        
        # Replace all NaN types with None using replace (more reliable)
        df_final = df_final.replace([np.nan, pd.NA, pd.NaT, float('nan')], None)
        df_final = df_final.replace(['nan', 'NaN', 'None'], None)
        
        # For any remaining NaN values, set them to None explicitly
        for col in df_final.columns:
            mask = df_final[col].isna()
            if mask.any():
                df_final.loc[mask, col] = None
        
        # Convert to dict
        results = df_final.to_dict('records')
        
        # Clean up any remaining NaN/Inf values in the dict (JSON can't serialize them)
        for record in results:
            for key, value in list(record.items()):
                # Check for NaN
                if value is None:
                    continue
                elif isinstance(value, float):
                    if math.isnan(value) or math.isinf(value):
                        record[key] = None
                elif isinstance(value, (np.floating, np.integer)):
                    if np.isnan(value) or np.isinf(value):
                        record[key] = None
                elif pd.isna(value):
                    record[key] = None
        
        update_task_status(
            task_id,
            'completed',
            'Pipeline completed successfully!',
            100,
            {
                'total_domains': len(results),
                'live_domains': len(live_domains),
                'candidates': len(variants),
                'results_file': str(dataset_file.relative_to(PROJECT_ROOT))
            }
        )
        task_results[task_id] = results
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        update_task_status(task_id, 'error', f'Pipeline failed: {error_msg}', 0)


def check_dns_live(candidates_file, progress_callback=None):
    """Check which domains are live (DNS resolution)"""
    import socket
    import threading
    from queue import Queue
    
    live_domains = set()
    queue = Queue()
    NUM_THREADS = 50
    
    def worker():
        while True:
            domain = queue.get()
            if domain is None:
                break
            try:
                socket.gethostbyname(domain)
                live_domains.add(domain)
            except:
                pass
            finally:
                queue.task_done()
                if progress_callback:
                    progress_callback(len(live_domains))
    
    # Read domains
    with open(candidates_file, 'r', encoding='utf-8') as f:
        domains = [line.strip() for line in f if line.strip()]
    
    # Start worker threads
    threads = []
    for _ in range(NUM_THREADS):
        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()
        threads.append(t)
    
    # Enqueue domains
    for d in domains:
        queue.put(d)
    
    # Wait for completion
    queue.join()
    
    # Stop threads
    for _ in threads:
        queue.put(None)
    for t in threads:
        t.join()
    
    return sorted(list(live_domains))




def compile_final_dataset(http_file):
    """Compile final dataset with screenshots"""
    df = pd.read_csv(http_file, encoding='utf-8')
    
    # Add screenshot paths
    screenshot_paths = []
    for domain in df['domain']:
        file_path = SCREENSHOTS_DIR / f"{domain.replace('.','_')}.png"
        if file_path.exists():
            screenshot_paths.append(str(file_path.relative_to(PROJECT_ROOT)))
        else:
            screenshot_paths.append('')
    
    df['screenshot'] = screenshot_paths
    
    output_file = SRC_DIR / "dataset_final.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')
    return output_file


@app.route('/api/results/<task_id>')
def get_results(task_id):
    """Get results for a completed task"""
    if task_id in task_results:
        return jsonify(task_results[task_id])
    return jsonify({'error': 'Results not found'}), 404


@app.route('/api/results/file/<filename>')
def get_results_file(filename):
    """Download results file"""
    file_path = SRC_DIR / filename
    if file_path.exists():
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404


@app.route('/api/logo-detection/download/<task_id>')
def download_logo_report(task_id):
    """Download logo detection HTML report"""
    try:
        # Try to get results from logo_results first
        if task_id in logo_results:
            logo_result = logo_results[task_id]
            results_data = logo_result.get('results', [])
            domain = logo_result.get('brand_domain', 'report')
        # Fallback: try to get from task_results
        elif task_id in task_results:
            results_data = task_results[task_id]
            # Check if this is logo detection results
            if results_data and len(results_data) > 0 and 'logo_detected' in results_data[0]:
                # Extract domain from first result or use task_id
                domain = results_data[0].get('domain', task_id).split('.')[-2] if '.' in results_data[0].get('domain', '') else task_id
            else:
                return jsonify({'error': 'Task found but not logo detection results'}), 404
        else:
            return jsonify({'error': f'Report not found for task {task_id}'}), 404
        
        # Generate report path
        domain_clean = domain.replace('.', '_').replace('/', '_')
        report_path = SRC_DIR / f"logo_detection_report_{domain_clean}.html"
        
        # Generate report if it doesn't exist
        if not report_path.exists():
            print(f"[INFO] Generating logo detection report for {domain}...")
            from logo_detection import LogoDetector
            LogoDetector.generate_report(results_data, domain, report_path)
        
        if report_path.exists():
            print(f"[INFO] Serving logo detection report: {report_path}")
            return send_file(
                str(report_path), 
                as_attachment=True, 
                download_name=f"logo_detection_report_{domain_clean}.html",
                mimetype='text/html'
            )
        else:
            print(f"[ERROR] Failed to generate report at {report_path}")
            return jsonify({'error': 'Failed to generate report file'}), 500
    except Exception as e:
        print(f"[ERROR] Error in download_logo_report: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/mimic-detection/download/<task_id>')
def download_mimic_report(task_id):
    """Download mimic detection HTML report"""
    try:
        # Try to get results from mimic_results first
        if task_id in mimic_results:
            mimic_result = mimic_results[task_id]
            results_data = mimic_result.get('results', [])
            domain = mimic_result.get('brand_domain', 'report')
        # Fallback: try to get from task_results
        elif task_id in task_results:
            results_data = task_results[task_id]
            # Check if this is mimic detection results
            if results_data and len(results_data) > 0 and 'mimic_brand' in results_data[0]:
                # Extract domain from first result or use task_id
                domain = results_data[0].get('domain', task_id).split('.')[-2] if '.' in results_data[0].get('domain', '') else task_id
            else:
                return jsonify({'error': 'Task found but not mimic detection results'}), 404
        else:
            return jsonify({'error': f'Report not found for task {task_id}'}), 404
        
        # Generate report path
        domain_clean = domain.replace('.', '_').replace('/', '_')
        report_path = SRC_DIR / f"mimic_detection_report_{domain_clean}.html"
        
        # Generate report if it doesn't exist
        if not report_path.exists():
            print(f"[INFO] Generating mimic detection report for {domain}...")
            from mimic_detection import MimicDetector
            MimicDetector.generate_report(results_data, domain, report_path)
        
        if report_path.exists():
            print(f"[INFO] Serving mimic detection report: {report_path}")
            return send_file(
                str(report_path), 
                as_attachment=True, 
                download_name=f"mimic_detection_report_{domain_clean}.html",
                mimetype='text/html'
            )
        else:
            print(f"[ERROR] Failed to generate report at {report_path}")
            return jsonify({'error': 'Failed to generate report file'}), 500
    except Exception as e:
        print(f"[ERROR] Error in download_mimic_report: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/threat-intelligence/results/<task_id>')
def get_ti_results(task_id):
    """Get threat intelligence results for a completed task"""
    if task_id in ti_results:
        return jsonify(ti_results[task_id])
    return jsonify({'error': 'Threat intelligence results not found'}), 404


@app.route('/api/threat-intelligence/download/<task_id>')
def download_ti_report(task_id):
    """Download threat intelligence report"""
    if task_id not in ti_results:
        return jsonify({'error': 'Report not found'}), 404
    
    try:
        ti_result = ti_results[task_id]
        domain = ti_result.get('domain', 'report')
        
        # Generate report if not exists
        report_path = SRC_DIR / f"threat_intelligence_report_{domain.replace('.', '_')}.html"
        if not report_path.exists():
            # Load API keys from environment variables
            api_keys = {
                'virustotal': os.getenv('VIRUSTOTAL_API_KEY', ''),
                'abuseipdb': os.getenv('ABUSEIPDB_API_KEY', ''),
                'safebrowsing': os.getenv('SAFEBROWSING_API_KEY', '')
            }
            collector = ThreatIntelligenceCollector(api_keys=api_keys)
            collector.generate_report(ti_result, report_path)
        
        return send_file(report_path, as_attachment=True, download_name=f"threat_intelligence_report_{domain}.html")
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dark-web/results/<task_id>')
def get_dw_results(task_id):
    """Get dark web monitoring results for a completed task"""
    if task_id in dw_results:
        return jsonify(dw_results[task_id])
    return jsonify({'error': 'Dark web monitoring results not found'}), 404


@app.route('/api/dark-web/download/<task_id>')
def download_dw_report(task_id):
    """Download dark web monitoring report"""
    if task_id not in dw_results:
        return jsonify({'error': 'Report not found'}), 404
    
    try:
        dw_result = dw_results[task_id]
        domain = dw_result.get('brand_domain', 'report')
        
        # Generate report if not exists
        report_path = SRC_DIR / f"dark_web_report_{domain.replace('.', '_')}.html"
        if not report_path.exists():
            # Load API keys from environment variables
            api_keys = {
                'intelx': os.getenv('INTELX_API_KEY', ''),
                'dehashed': os.getenv('DEHASHED_API_KEY', ''),
                'hibp': os.getenv('HIBP_API_KEY', '')
            }
            monitor = DarkWebMonitor(api_keys=api_keys)
            monitor.generate_report(dw_result, report_path)
        
        return send_file(report_path, as_attachment=True, download_name=f"dark_web_report_{domain}.html")
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/social-media/results/<task_id>')
def get_sm_results(task_id):
    """Get social media monitoring results for a completed task"""
    if task_id in sm_results:
        return jsonify(sm_results[task_id])
    return jsonify({'error': 'Social media monitoring results not found'}), 404


@app.route('/api/social-media/download/<task_id>')
def download_sm_report(task_id):
    """Download social media monitoring report"""
    if task_id not in sm_results:
        return jsonify({'error': 'Report not found'}), 404
    
    try:
        sm_result = sm_results[task_id]
        domain = sm_result.get('brand_domain', 'report')
        
        # Generate report if not exists
        report_path = SRC_DIR / f"social_media_report_{domain.replace('.', '_')}.html"
        if not report_path.exists():
            # Load API keys from environment variables
            api_keys = {
                'youtube': os.getenv('YOUTUBE_API_KEY', ''),
                'telegram': os.getenv('TELEGRAM_BOT_TOKEN', ''),
                'github': os.getenv('GITHUB_TOKEN', '')
            }
            monitor = SocialMediaMonitor(api_keys=api_keys)
            monitor.generate_report(sm_result, report_path)
        
        return send_file(report_path, as_attachment=True, download_name=f"social_media_report_{domain}.html")
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/screenshot/<path:filename>')
def get_screenshot(filename):
    """Serve screenshot images"""
    # Handle both screenshots and reference images
    if filename.startswith('screenshots/'):
        return send_from_directory(DATA_DIR, filename)
    elif filename.startswith('reference/'):
        return send_from_directory(DATA_DIR, filename)
    else:
        # Try screenshots first
        screenshot_path = SCREENSHOTS_DIR / filename
        if screenshot_path.exists():
            return send_from_directory(SCREENSHOTS_DIR, filename)
        # Try reference
        reference_path = REFERENCE_DIR / filename
        if reference_path.exists():
            return send_from_directory(REFERENCE_DIR, filename)
    
    return jsonify({'error': 'Screenshot not found'}), 404


@app.route('/api/datasets')
def list_datasets():
    """List available dataset files"""
    datasets = []
    for file in SRC_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(file, encoding='utf-8')
            datasets.append({
                'filename': file.name,
                'rows': len(df),
                'columns': list(df.columns),
                'size': file.stat().st_size,
                'modified': datetime.fromtimestamp(file.stat().st_mtime).isoformat()
            })
        except:
            pass
    
    return jsonify(datasets)


@app.route('/api/dataset/<filename>')
def get_dataset(filename):
    """Get dataset data"""
    file_path = SRC_DIR / filename
    if not file_path.exists():
        return jsonify({'error': 'Dataset not found'}), 404
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        # Limit to first 1000 rows for performance
        df_limited = df.head(1000)
        return jsonify({
            'data': df_limited.to_dict('records'),
            'total_rows': len(df),
            'columns': list(df.columns)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Starting Brand Protection Web Application...")
    print("Open your browser to http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

