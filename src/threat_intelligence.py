#!/usr/bin/env python3
"""
Threat Intelligence Module
Queries multiple free threat intelligence sources to gather security information about domains.
"""

import requests
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urlparse, quote
import re


class ThreatIntelligenceCollector:
    """
    Collects threat intelligence from multiple free sources.
    """
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize threat intelligence collector.
        
        Args:
            api_keys: Dictionary of API keys for services that require them.
                     Keys: 'virustotal', 'abuseipdb', 'safebrowsing'
        """
        self.api_keys = api_keys or {}
        self.results = {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def collect_all(self, domain: str) -> Dict:
        """
        Collect threat intelligence from all available sources.
        
        Args:
            domain: Domain name to check (e.g., 'example.com')
            
        Returns:
            Dictionary containing aggregated threat intelligence results
        """
        domain = domain.strip().lower()
        if not domain:
            return {'error': 'Invalid domain'}
        
        # Remove protocol if present
        domain = domain.replace('http://', '').replace('https://', '').split('/')[0]
        
        print(f"[TI] Collecting threat intelligence for: {domain}")
        
        results = {
            'domain': domain,
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'risk_score': 0,
            'risk_level': 'Unknown',
            'threats_found': [],
            'summary': {}
        }
        
        # Query all sources (removed urlvoid, urlhaus, and phishtank as requested)
        sources = [
            ('virustotal', self.check_virustotal),
            ('abuseipdb', self.check_abuseipdb),
            ('safebrowsing', self.check_safebrowsing),
            ('whois', self.check_whois_info),
        ]
        
        for source_name, check_func in sources:
            try:
                print(f"[TI] Checking {source_name}...")
                source_result = check_func(domain)
                results['sources'][source_name] = source_result
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"[TI] Error checking {source_name}: {e}")
                results['sources'][source_name] = {'error': str(e), 'status': 'failed'}
        
        # Calculate risk score
        risk_data = self.calculate_risk_score(results)
        results['risk_score'] = risk_data['score']
        results['risk_level'] = risk_data['level']
        results['threats_found'] = risk_data['threats']
        results['summary'] = risk_data['summary']
        
        return results
    
    def check_virustotal(self, domain: str) -> Dict:
        """Check domain on VirusTotal (requires API key for full access)."""
        result = {'source': 'VirusTotal', 'status': 'checked'}
        
        api_key = self.api_keys.get('virustotal')
        if not api_key:
            result['status'] = 'no_api_key'
            result['message'] = 'VirusTotal API key not provided (free tier available)'
            return result
        
        try:
            url = f"https://www.virustotal.com/vtapi/v2/domain/report"
            params = {
                'apikey': api_key,
                'domain': domain
            }
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                result['status'] = 'success'
                result['detections'] = data.get('detected_urls', [])
                result['undetected'] = data.get('undetected_urls', [])
                result['detection_count'] = len(result['detections'])
                result['last_scan'] = data.get('scan_date')
                result['reputation'] = data.get('reputation', 0)
            elif response.status_code == 204:
                result['status'] = 'rate_limited'
                result['message'] = 'VirusTotal rate limit exceeded'
            else:
                result['status'] = 'error'
                result['message'] = f"HTTP {response.status_code}"
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result
    
    def check_abuseipdb(self, domain: str) -> Dict:
        """Check domain on AbuseIPDB (requires API key)."""
        result = {'source': 'AbuseIPDB', 'status': 'checked'}
        
        api_key = self.api_keys.get('abuseipdb')
        if not api_key:
            result['status'] = 'no_api_key'
            result['message'] = 'AbuseIPDB API key not provided'
            return result
        
        try:
            # First, resolve domain to IP
            import socket
            try:
                ip = socket.gethostbyname(domain)
            except Exception as e:
                result['status'] = 'error'
                result['message'] = f'Could not resolve domain to IP: {str(e)}'
                return result
            
            url = "https://api.abuseipdb.com/api/v2/check"
            headers = {
                'Key': api_key,
                'Accept': 'application/json'
            }
            params = {
                'ipAddress': ip,
                'maxAgeInDays': 90,
                'verbose': ''
            }
            response = self.session.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                result['status'] = 'success'
                ip_data = data.get('data', {})
                
                # Extract all available fields
                result['ip'] = ip
                result['abuse_confidence'] = ip_data.get('abuseConfidencePercentage', 0)
                result['usage_type'] = ip_data.get('usageType', 'Unknown')
                result['is_public'] = ip_data.get('isPublic', False)
                result['is_whitelisted'] = ip_data.get('isWhitelisted', False)
                result['country'] = ip_data.get('countryCode', 'Unknown')
                result['isp'] = ip_data.get('isp', 'Unknown')
                result['domain'] = ip_data.get('domain', 'Unknown')
                result['total_reports'] = ip_data.get('totalReports', 0)
                result['num_distinct_users'] = ip_data.get('numDistinctUsers', 0)
                result['last_reported_at'] = ip_data.get('lastReportedAt', 'Never')
                
                # Add summary message
                if result['abuse_confidence'] > 0:
                    result['summary'] = f"IP {ip} has {result['abuse_confidence']}% abuse confidence with {result['total_reports']} total reports"
                else:
                    result['summary'] = f"IP {ip} appears clean (0% abuse confidence)"
            elif response.status_code == 429:
                result['status'] = 'rate_limited'
                result['message'] = 'AbuseIPDB rate limit exceeded'
            else:
                result['status'] = 'error'
                try:
                    error_data = response.json()
                    result['message'] = error_data.get('errors', [{}])[0].get('detail', f"HTTP {response.status_code}")
                except:
                    result['message'] = f"HTTP {response.status_code}: {response.text[:100]}"
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()
        
        return result
    
    def check_safebrowsing(self, domain: str) -> Dict:
        """Check domain on Google Safe Browsing (requires API key)."""
        result = {'source': 'Google Safe Browsing', 'status': 'checked'}
        
        api_key = self.api_keys.get('safebrowsing')
        if not api_key:
            result['status'] = 'no_api_key'
            result['message'] = 'Google Safe Browsing API key not provided'
            return result
        
        try:
            url = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={api_key}"
            payload = {
                'client': {
                    'clientId': 'brand-protection',
                    'clientVersion': '1.0'
                },
                'threatInfo': {
                    'threatTypes': ['MALWARE', 'SOCIAL_ENGINEERING', 'UNWANTED_SOFTWARE'],
                    'platformTypes': ['ANY_PLATFORM'],
                    'threatEntryTypes': ['URL'],
                    'threatEntries': [{'url': f"http://{domain}"}, {'url': f"https://{domain}"}]
                }
            }
            response = self.session.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                result['status'] = 'success'
                matches = data.get('matches', [])
                
                if matches:
                    result['is_unsafe'] = True
                    result['threats'] = []
                    result['threat_details'] = []
                    for match in matches:
                        threat_type = match.get('threatType', 'UNKNOWN')
                        platform_type = match.get('platformType', 'ANY_PLATFORM')
                        threat_entry_type = match.get('threatEntryType', 'URL')
                        result['threats'].append(threat_type)
                        result['threat_details'].append({
                            'threat_type': threat_type,
                            'platform': platform_type,
                            'entry_type': threat_entry_type
                        })
                    result['summary'] = f"Domain flagged as unsafe: {', '.join(set(result['threats']))}"
                else:
                    result['is_unsafe'] = False
                    result['threats'] = []
                    result['threat_details'] = []
                    result['summary'] = "Domain appears safe according to Google Safe Browsing"
            elif response.status_code == 400:
                result['status'] = 'error'
                try:
                    error_data = response.json()
                    result['message'] = error_data.get('error', {}).get('message', 'Invalid request')
                except:
                    result['message'] = 'Invalid API request'
            else:
                result['status'] = 'error'
                result['message'] = f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()
        
        return result
    
    def check_whois_info(self, domain: str) -> Dict:
        """Get basic WHOIS information (using python-whois or socket)."""
        result = {'source': 'WHOIS', 'status': 'checked'}
        
        try:
            # Try using python-whois library if available
            try:
                import whois
                w = whois.whois(domain)
                
                result['status'] = 'success'
                result['registrar'] = w.registrar if hasattr(w, 'registrar') and w.registrar else 'Unknown'
                result['created_date'] = str(w.creation_date[0]) if hasattr(w, 'creation_date') and w.creation_date else 'Unknown'
                result['expires_date'] = str(w.expiration_date[0]) if hasattr(w, 'expiration_date') and w.expiration_date else 'Unknown'
                result['updated_date'] = str(w.updated_date[0]) if hasattr(w, 'updated_date') and w.updated_date else 'Unknown'
                result['name_servers'] = list(w.name_servers) if hasattr(w, 'name_servers') and w.name_servers else []
                result['country'] = w.country if hasattr(w, 'country') and w.country else 'Unknown'
                result['org'] = w.org if hasattr(w, 'org') and w.org else 'Unknown'
                
                # Calculate domain age
                if result['created_date'] != 'Unknown':
                    try:
                        from dateutil import parser
                        created = parser.parse(result['created_date'])
                        days_old = (datetime.now() - created.replace(tzinfo=None)).days
                        result['domain_age_days'] = days_old
                        result['summary'] = f"Domain registered {days_old} days ago"
                    except:
                        result['summary'] = f"Domain created: {result['created_date']}"
                else:
                    result['summary'] = "WHOIS data retrieved successfully"
                    
            except ImportError:
                # Fallback: Use socket to get basic info
                import socket
                try:
                    ip = socket.gethostbyname(domain)
                    result['status'] = 'partial'
                    result['ip'] = ip
                    result['message'] = 'python-whois not installed. Install with: pip install python-whois'
                    result['summary'] = f"Domain resolves to IP: {ip}"
                except Exception as e:
                    result['status'] = 'error'
                    result['message'] = f'Could not resolve domain: {str(e)}'
            except Exception as e:
                result['status'] = 'partial'
                result['message'] = f'Limited WHOIS data: {str(e)}'
                # Try socket as fallback
                try:
                    import socket
                    ip = socket.gethostbyname(domain)
                    result['ip'] = ip
                    result['summary'] = f"Domain resolves to IP: {ip}"
                except:
                    pass
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result
    
    def calculate_risk_score(self, results: Dict) -> Dict:
        """
        Calculate overall risk score based on all threat intelligence sources.
        
        Returns:
            Dictionary with risk_score (0-100), risk_level, and threats list
        """
        risk_score = 0
        threats = []
        summary = {
            'total_sources': 0,
            'sources_with_threats': 0,
            'high_risk_indicators': 0,
            'medium_risk_indicators': 0,
            'low_risk_indicators': 0
        }
        
        sources = results.get('sources', {})
        
        # VirusTotal
        vt = sources.get('virustotal', {})
        if vt.get('status') == 'success':
            summary['total_sources'] += 1
            detections = vt.get('detection_count', 0)
            if detections > 0:
                summary['sources_with_threats'] += 1
                threats.append(f"VirusTotal: {detections} detections")
                risk_score += min(detections * 5, 30)  # Max 30 points
                summary['high_risk_indicators'] += 1
        
        # AbuseIPDB
        abuse = sources.get('abuseipdb', {})
        if abuse.get('status') == 'success':
            summary['total_sources'] += 1
            confidence = abuse.get('abuse_confidence', 0)
            if confidence > 50:
                summary['sources_with_threats'] += 1
                threats.append(f"AbuseIPDB: {confidence}% abuse confidence")
                risk_score += min(confidence / 2, 25)  # Max 25 points
                summary['high_risk_indicators'] += 1
            elif confidence > 25:
                risk_score += 10
                summary['medium_risk_indicators'] += 1
        
        # Safe Browsing
        safebrowsing = sources.get('safebrowsing', {})
        if safebrowsing.get('status') == 'success':
            summary['total_sources'] += 1
            if safebrowsing.get('is_unsafe', False):
                summary['sources_with_threats'] += 1
                threat_types = safebrowsing.get('threats', [])
                threats.append(f"Google Safe Browsing: {', '.join(threat_types)}")
                risk_score += 30  # High risk
                summary['high_risk_indicators'] += 1
            else:
                # Even if safe, count as checked source
                pass
        
        # WHOIS (domain age and suspicious patterns)
        whois = sources.get('whois', {})
        if whois.get('status') == 'success':
            summary['total_sources'] += 1
            created_date = whois.get('created_date', '')
            if created_date and created_date != 'Unknown':
                try:
                    from dateutil import parser
                    created = parser.parse(created_date)
                    days_old = (datetime.now() - created.replace(tzinfo=None)).days
                    if days_old < 30:
                        risk_score += 5  # New domain
                        summary['low_risk_indicators'] += 1
                        threats.append(f"WHOIS: Domain is only {days_old} days old")
                except:
                    pass
        
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
        Generate a formatted threat intelligence report.
        
        Args:
            results: Threat intelligence results dictionary
            output_path: Optional path to save report file
            
        Returns:
            Report as string (HTML format)
        """
        domain = results.get('domain', 'Unknown')
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
    <title>Threat Intelligence Report - {domain}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
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
            color: #666;
            margin-bottom: 30px;
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 5px;
        }}
        .section h2 {{
            color: #333;
            margin-top: 0;
        }}
        .threat-item {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
            border-radius: 3px;
        }}
        .source-result {{
            margin: 15px 0;
            padding: 15px;
            background: white;
            border-radius: 5px;
            border-left: 4px solid #0d6efd;
        }}
        .status-success {{
            color: #198754;
            font-weight: bold;
        }}
        .status-error {{
            color: #dc3545;
        }}
        .status-no-key {{
            color: #6c757d;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
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
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Threat Intelligence Report</h1>
        <div style="text-align: center; margin: 20px 0;">
            <h2 style="color: #666;">Domain: <strong>{domain}</strong></h2>
            <p style="color: #999;">Generated: {timestamp}</p>
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
                    <td>Sources with Threats</td>
                    <td>{summary.get('sources_with_threats', 0)}</td>
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
            {''.join([f'<div class="threat-item">{threat}</div>' for threat in threats]) if threats else '<p style="color: #198754;">No threats identified.</p>'}
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
                    if key not in ['source', 'status', 'error', 'message']:
                        if isinstance(value, (list, dict)):
                            value = json.dumps(value, indent=2)
                        html += f"<p><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>"
            elif status == 'no_api_key':
                html += f"<p style='color: #6c757d;'>{source_data.get('message', 'API key not provided')}</p>"
            elif status == 'error':
                html += f"<p style='color: #dc3545;'>Error: {source_data.get('error', source_data.get('message', 'Unknown error'))}</p>"
            
            html += "</div>"
        
        html += """
        </div>
        
        <div class="footer">
            <p>Report generated by Brand Protection Threat Intelligence System</p>
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
            print(f"[TI] Report saved to: {output_path}")
        
        return html


def collect_threat_intelligence(domain: str, api_keys: Optional[Dict[str, str]] = None) -> Dict:
    """
    Convenience function to collect threat intelligence for a domain.
    
    Args:
        domain: Domain name to check
        api_keys: Optional dictionary of API keys
        
    Returns:
        Threat intelligence results dictionary
    """
    collector = ThreatIntelligenceCollector(api_keys=api_keys)
    return collector.collect_all(domain)


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python threat_intelligence.py <domain> [api_keys.json]")
        sys.exit(1)
    
    domain = sys.argv[1]
    api_keys = None
    
    if len(sys.argv) > 2:
        api_keys_file = Path(sys.argv[2])
        if api_keys_file.exists():
            with open(api_keys_file, 'r') as f:
                api_keys = json.load(f)
    
    results = collect_threat_intelligence(domain, api_keys)
    
    print("\n" + "="*60)
    print("THREAT INTELLIGENCE RESULTS")
    print("="*60)
    print(f"Domain: {results.get('domain')}")
    print(f"Risk Score: {results.get('risk_score')}/100")
    print(f"Risk Level: {results.get('risk_level')}")
    print(f"\nThreats Found: {len(results.get('threats_found', []))}")
    for threat in results.get('threats_found', []):
        print(f"  - {threat}")
    
    # Generate report
    report_path = Path(f"threat_intelligence_report_{domain.replace('.', '_')}.html")
    collector = ThreatIntelligenceCollector(api_keys=api_keys)
    collector.generate_report(results, report_path)
    print(f"\nReport saved to: {report_path}")

