# http_info.py
#!/usr/bin/env python3
import requests
import socket
import ssl
import argparse
import csv
import urllib3

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_ip(domain):
    try:
        return socket.gethostbyname(domain)
    except:
        return None

def get_ssl_info(domain, final_url=None):
    """Get SSL certificate information - tries multiple methods"""
    ssl_data = {
        'issuer': '',
        'notBefore': '',
        'notAfter': ''
    }
    
    # Extract domain from final_url if provided and it's HTTPS
    target_domain = domain
    if final_url and final_url.startswith('https://'):
        try:
            from urllib.parse import urlparse
            parsed = urlparse(final_url)
            target_domain = parsed.netloc
        except:
            pass
    
    # Method 1: Try direct HTTPS connection to domain
    for check_domain in [target_domain, domain]:
        try:
            ctx = ssl.create_default_context()
            with ctx.wrap_socket(socket.socket(), server_hostname=check_domain) as s:
                s.settimeout(5)
                s.connect((check_domain, 443))
                cert = s.getpeercert()
                issuer = dict(x[0] for x in cert.get('issuer', []))
                if not ssl_data['issuer']:
                    ssl_data['issuer'] = issuer.get('organizationName', '') or issuer.get('commonName', '') or issuer.get('O', '')
                if not ssl_data['notBefore']:
                    ssl_data['notBefore'] = cert.get('notBefore', '')
                if not ssl_data['notAfter']:
                    ssl_data['notAfter'] = cert.get('notAfter', '')
                if ssl_data['issuer'] and ssl_data['notBefore'] and ssl_data['notAfter']:
                    return ssl_data
        except Exception:
            continue
    
    # Method 2: Try with requests using stream=True to get connection
    for check_domain in [target_domain, domain]:
        try:
            url = f'https://{check_domain}'
            response = requests.get(url, timeout=8, verify=False, stream=True, allow_redirects=False)
            # Close immediately to get connection info
            response.close()
            
            # Try to get cert from connection
            if hasattr(response, 'raw') and hasattr(response.raw, 'connection'):
                conn = response.raw.connection
                if hasattr(conn, 'sock'):
                    sock = conn.sock
                    if hasattr(sock, 'getpeercert'):
                        cert = sock.getpeercert()
                        issuer = dict(x[0] for x in cert.get('issuer', []))
                        if not ssl_data['issuer']:
                            ssl_data['issuer'] = issuer.get('organizationName', '') or issuer.get('commonName', '') or issuer.get('O', '')
                        if not ssl_data['notBefore']:
                            ssl_data['notBefore'] = cert.get('notBefore', '')
                        if not ssl_data['notAfter']:
                            ssl_data['notAfter'] = cert.get('notAfter', '')
                        if ssl_data['issuer'] and ssl_data['notBefore'] and ssl_data['notAfter']:
                            return ssl_data
        except Exception:
            continue
    
    # Method 3: Try with requests.get and check response connection
    for check_domain in [target_domain, domain]:
        try:
            url = f'https://{check_domain}'
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter()
            session.mount('https://', adapter)
            response = session.get(url, timeout=8, verify=False, stream=True)
            
            # Try to access the underlying connection
            if hasattr(response.raw, '_connection'):
                conn = response.raw._connection
                if hasattr(conn, 'sock'):
                    sock = conn.sock
                    if hasattr(sock, 'getpeercert'):
                        cert = sock.getpeercert()
                        issuer = dict(x[0] for x in cert.get('issuer', []))
                        if not ssl_data['issuer']:
                            ssl_data['issuer'] = issuer.get('organizationName', '') or issuer.get('commonName', '') or issuer.get('O', '')
                        if not ssl_data['notBefore']:
                            ssl_data['notBefore'] = cert.get('notBefore', '')
                        if not ssl_data['notAfter']:
                            ssl_data['notAfter'] = cert.get('notAfter', '')
                        if ssl_data['issuer'] and ssl_data['notBefore'] and ssl_data['notAfter']:
                            response.close()
                            return ssl_data
            response.close()
        except Exception:
            continue
    
    return ssl_data

def get_http_info(domain):
    """Get HTTP information - tries HTTPS first, then HTTP"""
    http_data = {
        'final_url': '',
        'status_code': '',
        'title': ''
    }
    
    # Try HTTPS first, then HTTP
    for protocol in ['https', 'http']:
        try:
            url = f"{protocol}://{domain}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            r = requests.get(url, timeout=10, allow_redirects=True, verify=False, headers=headers)
            http_data['final_url'] = r.url
            http_data['status_code'] = r.status_code
            
            # Extract title with multiple methods
            title = extract_title(r.text)
            if not title:
                # Try alternative title extraction
                title = extract_title_alternative(r.text)
            http_data['title'] = title
            
            # If we got data, return it
            if http_data['status_code']:
                return http_data
        except requests.exceptions.SSLError:
            continue
        except requests.exceptions.Timeout:
            continue
        except requests.exceptions.ConnectionError:
            continue
        except Exception as e:
            continue
    
    return http_data

def extract_title(html):
    """Extract page title using regex"""
    import re
    if not html:
        return ''
    
    # Method 1: Standard title tag
    match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
    if match:
        title = match.group(1).strip()
        # Clean up title
        title = re.sub(r'\s+', ' ', title)  # Replace multiple spaces
        title = title.replace('\n', ' ').replace('\r', ' ')
        if title:
            return title[:500]  # Limit length
    
    return ''

def extract_title_alternative(html):
    """Alternative title extraction methods"""
    import re
    if not html:
        return ''
    
    # Try meta title
    match = re.search(r'<meta[^>]*property=["\']og:title["\'][^>]*content=["\']([^"\']+)["\']', html, re.IGNORECASE)
    if match:
        return match.group(1).strip()[:500]
    
    # Try h1 as fallback
    match = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.IGNORECASE | re.DOTALL)
    if match:
        title = re.sub(r'<[^>]+>', '', match.group(1)).strip()
        if title:
            return title[:500]
    
    return ''

def is_real_brand(domain, final_url, title, brand_domain):
    """
    Check if domain is real brand (redirects to real website or is the real website).
    
    Args:
        domain: The domain being checked
        final_url: The final URL after redirects
        title: Page title
        brand_domain: The original brand domain (e.g., "facebook.com")
    
    Returns:
        "yes" or "no"
    """
    if not brand_domain:
        return "no"
    
    try:
        from urllib.parse import urlparse
        
        # Clean domains
        domain_clean = domain.lower().replace('www.', '').replace('http://', '').replace('https://', '')
        brand_clean = brand_domain.lower().replace('www.', '').replace('http://', '').replace('https://', '')
        
        # Check 1: Is it the exact brand domain?
        if domain_clean == brand_clean:
            return "yes"
        
        # Check 2: Does final URL point to brand domain?
        if final_url:
            parsed = urlparse(final_url.lower())
            final_domain = parsed.netloc.lower().replace('www.', '')
            
            if brand_clean == final_domain or brand_clean in final_domain:
                return "yes"
            
            # Check if brand domain is in the URL path
            if brand_clean in final_url.lower():
                return "yes"
        
        # Check 3: Does title match brand?
        if title:
            title_lower = title.lower()
            brand_name = brand_clean.split('.')[0]  # Extract "facebook" from "facebook.com"
            
            # Check if brand name appears prominently in title
            if brand_name in title_lower:
                # Additional check: title should be reasonable length and contain brand
                if len(title) < 200:
                    title_words = title_lower.split()
                    if brand_name in title_words or any(brand_name in word for word in title_words):
                        return "yes"
        
        return "no"
        
    except Exception as e:
        return "no"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="TXT file with domains")
    parser.add_argument("--output_file", default="http_features.csv")
    args = parser.parse_args()
    
    # Ask user for real brand domain (mandatory)
    print("\n" + "="*60)
    print("BRAND PROTECTION - HTTP INFO COLLECTOR")
    print("="*60)
    brand_domain = input("Enter the real brand domain (e.g., facebook.com): ").strip()
    
    if not brand_domain:
        print("[ERROR] Brand domain is required!")
        return
    
    # Clean brand domain
    brand_domain = brand_domain.lower().replace('http://', '').replace('https://', '').replace('www.', '')
    print(f"[INFO] Using brand domain: {brand_domain}")
    print("="*60 + "\n")

    from pathlib import Path
    
    # Handle file paths properly
    input_path = Path(args.input_file)
    if not input_path.is_absolute():
        input_path = Path(__file__).parent / args.input_file
    
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return
    
    # Read domains from input file (live_domains.txt)
    domains = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                domains.append(line)

    output_path = Path(__file__).parent / args.output_file
    
    # Fieldnames with real_brand column (mandatory)
    fieldnames = ['domain', 'ip', 'http_status', 'final_url', 'title', 
                  'ssl_issuer', 'ssl_notBefore', 'ssl_notAfter', 'real_brand']
    
    # Track valid domains and removed domains
    valid_domains = []
    removed_count = 0
    
    # Overwrite CSV file completely (not append)
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        csvfile.flush()  # Ensure header is written immediately

        total = len(domains)
        for idx, d in enumerate(domains, 1):
            print(f"[{idx}/{total}] Processing {d}...")
            try:
                # Get HTTP information first
                http_info = get_http_info(d)
                
                # Extract values with defaults
                final_url = http_info.get('final_url', '') if http_info else ''
                title = http_info.get('title', '') if http_info else ''
                status_code = http_info.get('status_code', '') if http_info else ''
                
                # If no final_url, remove from live_domains.txt immediately and skip
                if not final_url or final_url == '':
                    print(f"  [REMOVE] No final URL found - removing from live_domains.txt")
                    
                    # Remove from live_domains.txt in real-time
                    domains.remove(d)
                    removed_count += 1
                    
                    # Update live_domains.txt file immediately
                    with open(input_path, 'w', encoding='utf-8') as f:
                        for domain in domains:
                            f.write(domain + '\n')
                    
                    print(f"  [UPDATED] live_domains.txt - {len(domains)} domains remaining")
                    continue
                
                # Get IP
                ip = get_ip(d)
                
                # Get SSL info (pass final_url to help with extraction)
                ssl_info = get_ssl_info(d, final_url)
                
                # Check if real brand
                real_brand = is_real_brand(d, final_url, title, brand_domain)
                
                # Build row with all data
                row = {
                    'domain': d,
                    'ip': ip or '',
                    'http_status': str(status_code) if status_code else '',
                    'final_url': final_url,
                    'title': title,
                    'ssl_issuer': ssl_info.get('issuer', '') if ssl_info else '',
                    'ssl_notBefore': ssl_info.get('notBefore', '') if ssl_info else '',
                    'ssl_notAfter': ssl_info.get('notAfter', '') if ssl_info else '',
                    'real_brand': real_brand
                }
                
                # Write row to CSV immediately
                writer.writerow(row)
                csvfile.flush()  # Ensure data is written immediately
                valid_domains.append(d)
                
            except Exception as e:
                print(f"[ERROR] Failed to process {d}: {e}")
                # If error, also remove from live_domains.txt
                if d in domains:
                    domains.remove(d)
                    removed_count += 1
                    
                    # Update live_domains.txt file immediately
                    with open(input_path, 'w', encoding='utf-8') as f:
                        for domain in domains:
                            f.write(domain + '\n')
                    
                    print(f"  [REMOVED] Domain removed from live_domains.txt due to error")

    print(f"\n[DONE] Saved HTTP features to {output_path}")
    print(f"[INFO] Total valid domains (with final_url): {len(valid_domains)}")
    print(f"[INFO] Total removed domains (no final_url/errors): {removed_count}")
    print(f"[INFO] Final live_domains.txt contains: {len(domains)} domains")

if __name__ == "__main__":
    main()

