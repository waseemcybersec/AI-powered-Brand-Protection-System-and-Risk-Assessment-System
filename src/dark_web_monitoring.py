#!/usr/bin/env python3
"""
Dark Web Monitoring Module
Monitors dark web sources for brand mentions, credential leaks, and data breaches.
"""

import requests
import time
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import quote


class DarkWebMonitor:
    """
    Monitors dark web sources for brand security threats.
    """
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize dark web monitor.
        
        Args:
            api_keys: Dictionary of API keys for services that require them.
                     Keys: 'intelx', 'dehashed', 'hibp'
                     For DeHashed: can also include 'dehashed_email' for the account email
        """
        self.api_keys = api_keys or {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def monitor_brand(self, brand_domain: str, brand_name: Optional[str] = None) -> Dict:
        """
        Monitor dark web sources for brand-related threats.
        
        Args:
            brand_domain: Domain name to monitor (e.g., 'example.com')
            brand_name: Optional brand name to search for
            
        Returns:
            Dictionary containing dark web monitoring results
        """
        brand_domain = brand_domain.strip().lower()
        if not brand_domain:
            return {'error': 'Invalid domain'}
        
        # Remove protocol if present
        brand_domain = brand_domain.replace('http://', '').replace('https://', '').split('/')[0]
        
        # Extract brand name from domain if not provided
        if not brand_name:
            brand_name = brand_domain.split('.')[0]
        
        print(f"[Dark Web] Monitoring dark web for: {brand_domain} (brand: {brand_name})")
        
        results = {
            'brand_domain': brand_domain,
            'brand_name': brand_name,
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'risk_score': 0,
            'risk_level': 'Unknown',
            'threats_found': [],
            'summary': {}
        }
        
        # Query all sources
        sources = [
            ('intelx', self.check_intelx),
            ('dehashed', self.check_dehashed),
            ('hibp', self.check_hibp),
            ('paste_sites', self.check_paste_sites),
        ]
        
        for source_name, check_func in sources:
            try:
                print(f"[Dark Web] Checking {source_name}...")
                source_result = check_func(brand_domain, brand_name)
                results['sources'][source_name] = source_result
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"[Dark Web] Error checking {source_name}: {e}")
                results['sources'][source_name] = {'error': str(e), 'status': 'failed'}
        
        # Calculate risk score
        risk_data = self.calculate_risk_score(results)
        results['risk_score'] = risk_data['score']
        results['risk_level'] = risk_data['level']
        results['threats_found'] = risk_data['threats']
        results['summary'] = risk_data['summary']
        
        return results
    
    def check_intelx(self, domain: str, brand_name: str) -> Dict:
        """Check IntelligenceX for dark web mentions."""
        result = {'source': 'IntelligenceX', 'status': 'checked'}
        
        api_key = self.api_keys.get('intelx', '').strip()
        if not api_key:
            result['status'] = 'no_api_key'
            result['message'] = 'IntelligenceX API key not provided'
            return result
        
        # Debug: Log API key (first and last few chars only for security)
        print(f"[Dark Web] IntelligenceX API key: {api_key[:5]}...{api_key[-5:] if len(api_key) > 10 else 'short'} (length: {len(api_key)})")
        
        try:
            # IntelligenceX API - try multiple endpoints and auth methods
            endpoints_to_try = [
                {
                    'url': "https://free.intelx.io/phonebook/search",
                    'auth': 'url_param',
                    'name': 'free.intelx.io'
                },
                {
                    'url': "https://2.intelx.io/phonebook/search",
                    'auth': 'header',
                    'name': '2.intelx.io'
                },
                {
                    'url': "https://intelx.io/phonebook/search",
                    'auth': 'header',
                    'name': 'intelx.io'
                }
            ]
            
            search_payload = {
                'term': domain,
                'maxresults': 50,
                'media': 0,  # All media types
                'target': 0,  # All targets
                'terminate': []
            }
            
            response = None
            last_error = None
            successful_endpoint = None
            
            # Try each endpoint with corresponding auth method
            for endpoint_info in endpoints_to_try:
                endpoint = endpoint_info['url']
                auth_method = endpoint_info['auth']
                endpoint_name = endpoint_info.get('name', endpoint)
                try:
                    if auth_method == "header":
                        headers = {
                            'x-key': api_key,
                            'Content-Type': 'application/json',
                            'User-Agent': 'BrandProtection-System/1.0'
                        }
                        print(f"[Dark Web] Trying IntelligenceX endpoint (header auth): {endpoint_name}")
                        response = self.session.post(endpoint, json=search_payload, headers=headers, timeout=15)
                    elif auth_method == "url_param":
                        headers = {
                            'Content-Type': 'application/json',
                            'User-Agent': 'BrandProtection-System/1.0'
                        }
                        url_with_key = f"{endpoint}?k={api_key}"
                        print(f"[Dark Web] Trying IntelligenceX endpoint (URL param auth): {endpoint_name}")
                        response = self.session.post(url_with_key, json=search_payload, headers=headers, timeout=15)
                    
                    print(f"[Dark Web] Response status: {response.status_code}")
                    if response.status_code == 200:
                        successful_endpoint = endpoint
                        break  # Success!
                    elif response.status_code == 401:
                        print(f"[Dark Web] 401 Unauthorized - API key might be invalid")
                        print(f"[Dark Web] Response: {response.text[:300]}")
                        # Continue trying other endpoints
                    elif response.status_code != 404:
                        print(f"[Dark Web] Got status {response.status_code}, stopping")
                        break  # Got a definitive error (not 404), stop trying
                except Exception as e:
                    last_error = str(e)
                    print(f"[Dark Web] Exception: {e}")
                    continue
            
            if response is None:
                raise Exception(f"All endpoints failed. Last error: {last_error}")
            
            if successful_endpoint:
                print(f"[Dark Web] Successfully connected to: {successful_endpoint}")
            
            if response.status_code == 200:
                data = response.json()
                result['status'] = 'success'
                result['total_results'] = data.get('total', 0)
                result['selectors'] = data.get('selectors', [])
                
                # Extract unique sources
                sources = set()
                for selector in result['selectors']:
                    source = selector.get('source', 'Unknown')
                    sources.add(source)
                
                result['unique_sources'] = list(sources)
                result['source_count'] = len(sources)
                
                if result['total_results'] > 0:
                    result['summary'] = f"Found {result['total_results']} mentions across {result['source_count']} sources"
                    result['has_findings'] = True
                else:
                    result['summary'] = "No dark web mentions found"
                    result['has_findings'] = False
            elif response.status_code == 401:
                result['status'] = 'error'
                result['message'] = f'Invalid IntelligenceX API key (401). API key used: {api_key[:10]}... (first 10 chars). Please verify your API key at https://intelx.io/'
                result['debug_info'] = f'Endpoint tried: {endpoint if "endpoint" in locals() else "multiple"}, Response: {response.text[:200]}'
            elif response.status_code == 404:
                result['status'] = 'error'
                result['message'] = 'IntelligenceX API endpoint not found (404). The endpoint may have changed. Please check IntelligenceX documentation.'
            else:
                result['status'] = 'error'
                try:
                    error_data = response.json()
                    result['message'] = f"HTTP {response.status_code}: {error_data.get('message', response.text[:200])}"
                except:
                    result['message'] = f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()
        
        return result
    
    def check_dehashed(self, domain: str, brand_name: str) -> Dict:
        """Check DeHashed for credential leaks."""
        result = {'source': 'DeHashed', 'status': 'checked'}
        
        api_key = self.api_keys.get('dehashed', '').strip()
        if not api_key:
            result['status'] = 'no_api_key'
            result['message'] = 'DeHashed API key not provided'
            return result
        
        # Get email if provided (DeHashed requires email:api_key for Basic Auth)
        dehashed_email = self.api_keys.get('dehashed_email', '').strip()
        
        # Debug: Log credentials (partial for security)
        print(f"[Dark Web] DeHashed Email: {dehashed_email}")
        print(f"[Dark Web] DeHashed API key: {api_key[:5]}...{api_key[-5:] if len(api_key) > 10 else 'short'} (length: {len(api_key)})")
        
        try:
            # DeHashed API uses Basic Authentication with email:api_key format
            # We'll try multiple endpoint variations and ensure query params are encoded properly
            endpoints_to_try = [
                "https://api.dehashed.com/search",
                "https://dehashed.com/api/v1/search",
                "https://dehashed.com/search",
                "https://api.dehashed.com/v1/search"
            ]
            
            headers = {
                'Accept': 'application/json',
                'User-Agent': 'BrandProtection-System/1.0'
            }
            
            from requests.auth import HTTPBasicAuth
            
            if not dehashed_email:
                result['status'] = 'error'
                result['message'] = 'DeHashed requires email address. Please add DEHASHED_EMAIL to .env file.'
                return result
            
            email = dehashed_email.strip()
            key = api_key.strip()
            auth = HTTPBasicAuth(email, key)
            params = {'query': f'domain:{domain}'}
            
            response = None
            last_error = None
            successful_endpoint = None
            
            for endpoint in endpoints_to_try:
                try:
                    print(f"[Dark Web] Trying DeHashed endpoint: {endpoint}")
                    response = self.session.get(endpoint, headers=headers, auth=auth, params=params, timeout=15)
                    print(f"[Dark Web] Response status: {response.status_code}")
                    if response.status_code == 200:
                        successful_endpoint = endpoint
                        break
                    elif response.status_code == 401:
                        print(f"[Dark Web] 401 Unauthorized - credentials might be invalid")
                        print(f"[Dark Web] Response: {response.text[:300]}")
                    elif response.status_code not in [404]:
                        print(f"[Dark Web] Got status {response.status_code}. Response: {response.text[:300]}")
                        break
                except Exception as e:
                    last_error = str(e)
                    print(f"[Dark Web] Exception: {e}")
                    continue
            
            if response is None:
                raise Exception(f"All endpoints failed. Last error: {last_error}")
            
            if successful_endpoint:
                print(f"[Dark Web] Successfully connected to: {successful_endpoint}")
            
            if response.status_code == 200:
                data = response.json()
                result['status'] = 'success'
                result['total'] = data.get('total', 0)
                result['entries'] = data.get('entries', [])
                
                # Extract unique breach sources
                breach_sources = set()
                email_count = 0
                password_count = 0
                
                for entry in result['entries']:
                    database = entry.get('database', 'Unknown')
                    breach_sources.add(database)
                    if entry.get('email'):
                        email_count += 1
                    if entry.get('password'):
                        password_count += 1
                
                result['breach_sources'] = list(breach_sources)
                result['unique_breaches'] = len(breach_sources)
                result['email_count'] = email_count
                result['password_count'] = password_count
                
                if result['total'] > 0:
                    result['summary'] = f"Found {result['total']} leaked records across {result['unique_breaches']} breach(es)"
                    result['has_findings'] = True
                else:
                    result['summary'] = "No credential leaks found"
                    result['has_findings'] = False
            elif response.status_code == 401:
                result['status'] = 'error'
                result['message'] = f'DeHashed authentication failed (401). Email: {email}, API Key: {key[:10]}... (first 10 chars). Please verify credentials at https://app.dehashed.com/'
                result['debug_info'] = f'Endpoint tried: {endpoint if "endpoint" in locals() else "multiple"}, Response: {response.text[:200]}'
            elif response.status_code == 404:
                result['status'] = 'error'
                result['message'] = f'DeHashed API endpoint not found (404). Tried multiple endpoints. Please check: https://app.dehashed.com/documentation/api'
                result['debug_info'] = f'Response: {response.text[:200]}'
            elif response.status_code == 429:
                result['status'] = 'rate_limited'
                result['message'] = 'DeHashed rate limit exceeded'
            else:
                result['status'] = 'error'
                result['message'] = f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()
        
        return result
    
    def check_hibp(self, domain: str, brand_name: str) -> Dict:
        """Check Have I Been Pwned for domain breaches."""
        result = {'source': 'Have I Been Pwned', 'status': 'checked'}
        
        api_key = self.api_keys.get('hibp')
        
        try:
            # Search for domain breaches
            url = f"https://haveibeenpwned.com/api/v3/breaches?domain={domain}"
            headers = {
                'User-Agent': 'BrandProtection-System'
            }
            
            # Add API key if available (for higher rate limits)
            if api_key:
                headers['hibp-api-key'] = api_key
            
            response = self.session.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                breaches = response.json()
                result['status'] = 'success'
                result['breach_count'] = len(breaches)
                result['breaches'] = []
                
                for breach in breaches:
                    breach_info = {
                        'name': breach.get('Name', 'Unknown'),
                        'domain': breach.get('Domain', ''),
                        'breach_date': breach.get('BreachDate', ''),
                        'added_date': breach.get('AddedDate', ''),
                        'modified_date': breach.get('ModifiedDate', ''),
                        'pwn_count': breach.get('PwnCount', 0),
                        'description': breach.get('Description', ''),
                        'data_classes': breach.get('DataClasses', [])
                    }
                    result['breaches'].append(breach_info)
                
                if result['breach_count'] > 0:
                    total_pwned = sum(b['pwn_count'] for b in result['breaches'])
                    result['total_accounts_pwned'] = total_pwned
                    result['summary'] = f"Domain found in {result['breach_count']} breach(es) affecting {total_pwned:,} accounts"
                    result['has_findings'] = True
                else:
                    result['summary'] = "No breaches found for this domain"
                    result['has_findings'] = False
            elif response.status_code == 429:
                result['status'] = 'rate_limited'
                result['message'] = 'Have I Been Pwned rate limit exceeded'
            else:
                result['status'] = 'error'
                result['message'] = f"HTTP {response.status_code}"
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result
    
    def check_paste_sites(self, domain: str, brand_name: str) -> Dict:
        """Check paste sites for brand mentions."""
        result = {'source': 'Paste Sites', 'status': 'checked'}
        
        try:
            # Search pastebin.com (public API)
            pastebin_url = "https://pastebin.com/api/api_post.php"
            # Note: Pastebin requires API key for search, so we'll use alternative method
            
            # Try searching via public pastebin archive sites
            # This is a simplified approach - in production, you'd want to use proper APIs
            mentions = []
            
            # Check paste.ee (if accessible)
            try:
                # This is a placeholder - actual implementation would require proper scraping
                # or API access
                result['status'] = 'success'
                result['mentions_found'] = 0
                result['summary'] = "Paste site monitoring requires API access or specialized scraping"
                result['has_findings'] = False
                result['note'] = "Paste site monitoring is limited without API access"
            except Exception as e:
                result['status'] = 'partial'
                result['message'] = f"Limited paste site access: {str(e)}"
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result
    
    def calculate_risk_score(self, results: Dict) -> Dict:
        """
        Calculate overall dark web risk score.
        
        Returns:
            Dictionary with risk_score (0-100), risk_level, and threats list
        """
        risk_score = 0
        threats = []
        summary = {
            'total_sources': 0,
            'sources_with_findings': 0,
            'high_risk_indicators': 0,
            'medium_risk_indicators': 0,
            'low_risk_indicators': 0
        }
        
        sources = results.get('sources', {})
        
        # IntelligenceX
        intelx = sources.get('intelx', {})
        if intelx.get('status') == 'success':
            summary['total_sources'] += 1
            total_results = intelx.get('total_results', 0)
            if total_results > 0:
                summary['sources_with_findings'] += 1
                source_count = intelx.get('source_count', 0)
                threats.append(f"IntelligenceX: {total_results} dark web mentions across {source_count} source(s)")
                risk_score += min(total_results * 2 + source_count * 5, 40)  # Max 40 points
                summary['high_risk_indicators'] += 1
        
        # DeHashed
        dehashed = sources.get('dehashed', {})
        if dehashed.get('status') == 'success':
            summary['total_sources'] += 1
            total = dehashed.get('total', 0)
            if total > 0:
                summary['sources_with_findings'] += 1
                unique_breaches = dehashed.get('unique_breaches', 0)
                email_count = dehashed.get('email_count', 0)
                password_count = dehashed.get('password_count', 0)
                threats.append(f"DeHashed: {total} leaked records ({email_count} emails, {password_count} passwords) across {unique_breaches} breach(es)")
                risk_score += min(total * 3 + unique_breaches * 10, 35)  # Max 35 points
                summary['high_risk_indicators'] += 1
        
        # Have I Been Pwned
        hibp = sources.get('hibp', {})
        if hibp.get('status') == 'success':
            summary['total_sources'] += 1
            breach_count = hibp.get('breach_count', 0)
            if breach_count > 0:
                summary['sources_with_findings'] += 1
                total_pwned = hibp.get('total_accounts_pwned', 0)
                threats.append(f"Have I Been Pwned: Domain in {breach_count} breach(es) affecting {total_pwned:,} accounts")
                risk_score += min(breach_count * 8 + (total_pwned // 1000), 30)  # Max 30 points
                summary['high_risk_indicators'] += 1
        
        # Paste Sites
        paste_sites = sources.get('paste_sites', {})
        if paste_sites.get('status') == 'success':
            summary['total_sources'] += 1
            mentions = paste_sites.get('mentions_found', 0)
            if mentions > 0:
                summary['sources_with_findings'] += 1
                threats.append(f"Paste Sites: {mentions} mention(s) found")
                risk_score += min(mentions * 3, 15)  # Max 15 points
                summary['medium_risk_indicators'] += 1
        
        # Normalize risk score to 0-100
        risk_score = min(risk_score, 100)
        
        # Determine risk level
        if risk_score >= 70:
            risk_level = 'Critical'
        elif risk_score >= 50:
            risk_level = 'High'
        elif risk_score >= 30:
            risk_level = 'Medium'
        elif risk_score >= 10:
            risk_level = 'Low'
        else:
            risk_level = 'Very Low'
        
        return {
            'score': round(risk_score, 2),
            'level': risk_level,
            'threats': threats,
            'summary': summary
        }
    
    def generate_report(self, results: Dict, output_path: Optional[Path] = None) -> str:
        """
        Generate a formatted dark web monitoring report.
        
        Args:
            results: Dark web monitoring results dictionary
            output_path: Optional path to save report file
            
        Returns:
            Report as string (HTML format)
        """
        domain = results.get('brand_domain', 'Unknown')
        brand_name = results.get('brand_name', 'Unknown')
        risk_score = results.get('risk_score', 0)
        risk_level = results.get('risk_level', 'Unknown')
        threats = results.get('threats_found', [])
        summary = results.get('summary', {})
        sources = results.get('sources', {})
        timestamp = results.get('timestamp', 'Unknown')
        
        # Determine risk color
        if risk_score >= 70:
            risk_color = '#dc3545'  # Red
        elif risk_score >= 50:
            risk_color = '#fd7e14'  # Orange
        elif risk_score >= 30:
            risk_color = '#ffc107'  # Yellow
        elif risk_score >= 10:
            risk_color = '#0dcaf0'  # Cyan
        else:
            risk_color = '#198754'  # Green
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dark Web Monitoring Report - {domain}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: #e0e0e0;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: #2d2d2d;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.5);
        }}
        h1 {{
            color: #fff;
            border-bottom: 3px solid {risk_color};
            padding-bottom: 10px;
        }}
        .risk-score {{
            font-size: 48px;
            font-weight: bold;
            color: {risk_color};
            text-align: center;
            margin: 20px 0;
        }}
        .risk-level {{
            text-align: center;
            font-size: 24px;
            color: #999;
            margin-bottom: 30px;
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            background: #1a1a1a;
            border-radius: 5px;
        }}
        .section h2 {{
            color: #fff;
            margin-top: 0;
        }}
        .threat-item {{
            background: #3d3d3d;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
            border-radius: 3px;
        }}
        .source-result {{
            margin: 15px 0;
            padding: 15px;
            background: #1a1a1a;
            border-radius: 5px;
            border-left: 4px solid #0d6efd;
        }}
        .status-success {{
            color: #4caf50;
            font-weight: bold;
        }}
        .status-error {{
            color: #f44336;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #444;
        }}
        th {{
            background-color: #0d6efd;
            color: white;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #444;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Dark Web Monitoring Report</h1>
        <div style="text-align: center; margin: 20px 0;">
            <h2 style="color: #999;">Brand: <strong>{brand_name}</strong> ({domain})</h2>
            <p style="color: #666;">Generated: {timestamp}</p>
        </div>
        
        <div class="risk-score">{risk_score}/100</div>
        <div class="risk-level">Risk Level: {risk_level}</div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Sources Checked</td>
                    <td>{summary.get('total_sources', 0)}</td>
                </tr>
                <tr>
                    <td>Sources with Findings</td>
                    <td>{summary.get('sources_with_findings', 0)}</td>
                </tr>
                <tr>
                    <td>High Risk Indicators</td>
                    <td>{summary.get('high_risk_indicators', 0)}</td>
                </tr>
                <tr>
                    <td>Medium Risk Indicators</td>
                    <td>{summary.get('medium_risk_indicators', 0)}</td>
                </tr>
                <tr>
                    <td>Low Risk Indicators</td>
                    <td>{summary.get('low_risk_indicators', 0)}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Threats Identified</h2>
            {''.join([f'<div class="threat-item">{threat}</div>' for threat in threats]) if threats else '<p style="color: #4caf50;">No threats identified.</p>'}
        </div>
        
        <div class="section">
            <h2>Source Details</h2>
"""
        
        # Add source results
        for source_name, source_data in sources.items():
            status = source_data.get('status', 'unknown')
            status_class = 'status-success' if status == 'success' else ('status-error' if status == 'error' else 'status-no-key')
            
            html += f"""
            <div class="source-result">
                <h3>{source_data.get('source', source_name.title())}</h3>
                <p><strong>Status:</strong> <span class="{status_class}">{status}</span></p>
"""
            
            if status == 'success':
                # Add key-value pairs
                for key, value in source_data.items():
                    if key not in ['source', 'status', 'error', 'message', 'traceback']:
                        if isinstance(value, (list, dict)):
                            value = json.dumps(value, indent=2)
                        html += f"<p><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>"
            elif status == 'no_api_key':
                html += f"<p style='color: #999;'>{source_data.get('message', 'API key not provided')}</p>"
            elif status == 'error':
                html += f"<p style='color: #f44336;'>Error: {source_data.get('error', source_data.get('message', 'Unknown error'))}</p>"
            
            html += "</div>"
        
        html += """
        </div>
        
        <div class="footer">
            <p>Report generated by Brand Protection Dark Web Monitoring System</p>
            <p>This report is for informational purposes only. Always verify threats through multiple sources.</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Save to file if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"[Dark Web] Report saved to: {output_path}")
        
        return html


def monitor_dark_web(brand_domain: str, brand_name: Optional[str] = None, api_keys: Optional[Dict[str, str]] = None) -> Dict:
    """
    Convenience function to monitor dark web for a brand.
    
    Args:
        brand_domain: Domain name to monitor
        brand_name: Optional brand name
        api_keys: Optional dictionary of API keys
        
    Returns:
        Dark web monitoring results dictionary
    """
    monitor = DarkWebMonitor(api_keys=api_keys)
    return monitor.monitor_brand(brand_domain, brand_name)


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dark_web_monitoring.py <domain> [brand_name] [api_keys.json]")
        sys.exit(1)
    
    domain = sys.argv[1]
    brand_name = sys.argv[2] if len(sys.argv) > 2 else None
    api_keys = None
    
    if len(sys.argv) > 3:
        api_keys_file = Path(sys.argv[3])
        if api_keys_file.exists():
            with open(api_keys_file, 'r') as f:
                api_keys = json.load(f)
    
    results = monitor_dark_web(domain, brand_name, api_keys)
    
    print("\n" + "="*60)
    print("DARK WEB MONITORING RESULTS")
    print("="*60)
    print(f"Brand: {results.get('brand_name')} ({results.get('brand_domain')})")
    print(f"Risk Score: {results.get('risk_score')}/100")
    print(f"Risk Level: {results.get('risk_level')}")
    print(f"\nThreats Found: {len(results.get('threats_found', []))}")
    for threat in results.get('threats_found', []):
        print(f"  - {threat}")
    
    # Generate report
    report_path = Path(f"dark_web_report_{domain.replace('.', '_')}.html")
    monitor = DarkWebMonitor(api_keys=api_keys)
    monitor.generate_report(results, report_path)
    print(f"\nReport saved to: {report_path}")

