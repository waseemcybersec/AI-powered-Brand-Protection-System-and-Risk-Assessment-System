#!/usr/bin/env python3
"""
Social Media Monitoring Module

Monitors multiple social media platforms for brand impersonation:
- YouTube: Fake channels, scam videos
- Telegram: Username impersonation, scam groups
- GitHub: Exposed secrets, phishing kits

Features:
- Generic threat detection for ANY brand
- Typosquatting variant generation
- Dynamic threat keyword templates
"""

import os
import re
import json
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from urllib.parse import quote


class TyposquatGenerator:
    """Generate typosquatting variants of a brand name"""
    
    @staticmethod
    def generate_variants(brand_name: str) -> Set[str]:
        """
        Generate common typosquatting variants of a brand name
        
        Args:
            brand_name: Original brand name (e.g., 'facebook')
        
        Returns:
            Set of typosquatting variants
        """
        variants = {brand_name.lower()}
        name = brand_name.lower()
        
        # Character substitutions (common typos and look-alikes)
        substitutions = {
            'a': ['4', '@', 'aa'],
            'e': ['3', 'ee'],
            'i': ['1', 'l', '!', 'ii'],
            'o': ['0', 'oo'],
            's': ['5', '$', 'ss'],
            't': ['7', '+'],
            'l': ['1', 'i', '|'],
            'b': ['8', '6'],
            'g': ['9', 'q'],
        }
        
        # Apply substitutions
        for char, subs in substitutions.items():
            if char in name:
                for sub in subs:
                    variants.add(name.replace(char, sub, 1))
        
        # Missing characters (omission)
        for i in range(len(name)):
            variants.add(name[:i] + name[i+1:])
        
        # Doubled characters
        for i in range(len(name)):
            variants.add(name[:i] + name[i] + name[i:])
        
        # Adjacent character swaps
        for i in range(len(name) - 1):
            swapped = list(name)
            swapped[i], swapped[i+1] = swapped[i+1], swapped[i]
            variants.add(''.join(swapped))
        
        # Common prefix/suffix additions
        prefixes = ['get', 'my', 'the', 'real', 'official', 'true', 'legit']
        suffixes = ['app', 'official', 'hq', 'inc', 'corp', 'co', 'online', 'web']
        
        for prefix in prefixes:
            variants.add(f"{prefix}{name}")
            variants.add(f"{prefix}_{name}")
        
        for suffix in suffixes:
            variants.add(f"{name}{suffix}")
            variants.add(f"{name}_{suffix}")
        
        # Remove very short variants (less than 3 chars)
        variants = {v for v in variants if len(v) >= 3}
        
        return variants


class ThreatKeywordGenerator:
    """Generate threat-related search keywords for any brand"""
    
    # Threat keyword templates - {brand} will be replaced with actual brand name
    YOUTUBE_THREAT_TEMPLATES = [
        # Credential theft / Hacking
        "{brand} hack account",
        "{brand} password hack",
        "{brand} account hack 2024",
        "{brand} login hack",
        "{brand} crack password",
        "{brand} steal account",
        
        # Phishing
        "{brand} phishing page",
        "{brand} phishing tutorial",
        "{brand} fake login",
        "{brand} clone login page",
        
        # Scams
        "{brand} free money",
        "{brand} unlimited free",
        "{brand} generator 2024",
        "{brand} gift card generator",
        "{brand} free premium",
        
        # Security bypass
        "{brand} bypass verification",
        "{brand} bypass 2fa",
        "{brand} otp bypass",
        
        # Malware/Exploits
        "{brand} exploit",
        "{brand} vulnerability",
        "{brand} malware",
    ]
    
    GITHUB_CODE_TEMPLATES = [
        # Exposed secrets
        '"{brand}_api_key"',
        '"{brand}_secret"',
        '"{brand}_token"',
        '"{brand}_password"',
        '"{brand}_credentials"',
        '"api_key_{brand}"',
        '"secret_{brand}"',
        
        # Email patterns
        '"@{domain}"',
        
        # Phishing kits
        '"{brand} phishing"',
        '"{brand} login" html',
        '"{brand} fake page"',
    ]
    
    GITHUB_REPO_TEMPLATES = [
        "{brand} phishing",
        "{brand} phish",
        "{brand} fake login",
        "{brand} clone",
        "{brand} scam",
        "{brand} hack",
        "{brand} exploit",
        "{brand} credential",
    ]
    
    TELEGRAM_USERNAME_TEMPLATES = [
        # Fake support (very common scam)
        "{brand}_support",
        "{brand}support",
        "{brand}_helpdesk",
        "{brand}_help",
        "{brand}_customer",
        "{brand}_customercare",
        "{brand}_service",
        
        # Giveaway scams
        "{brand}_giveaway",
        "{brand}_giveaways",
        "{brand}_airdrop",
        "{brand}_free",
        "{brand}_prize",
        "{brand}_rewards",
        
        # Recovery scams
        "{brand}_recovery",
        "{brand}_restore",
        "{brand}_unlock",
        
        # Verification scams
        "{brand}_verify",
        "{brand}_verification",
        "{brand}_verified",
        
        # Admin impersonation
        "{brand}_admin",
        "{brand}_mod",
        "{brand}_moderator",
        "{brand}_team",
        "{brand}_staff",
        
        # Official impersonation
        "{brand}_official",
        "official_{brand}",
        "real_{brand}",
        "the_{brand}",
    ]
    
    @classmethod
    def generate_youtube_queries(cls, brand_name: str, include_typos: bool = True) -> List[str]:
        """Generate YouTube threat search queries"""
        queries = []
        brands_to_check = [brand_name.lower()]
        
        if include_typos:
            # Add top typosquatting variants (limit to avoid too many API calls)
            typos = list(TyposquatGenerator.generate_variants(brand_name))[:5]
            brands_to_check.extend(typos)
        
        for brand in brands_to_check:
            for template in cls.YOUTUBE_THREAT_TEMPLATES:
                queries.append(template.format(brand=brand))
        
        return queries
    
    @classmethod
    def generate_github_code_queries(cls, brand_name: str, brand_domain: str, include_typos: bool = True) -> List[str]:
        """Generate GitHub code search queries"""
        queries = []
        brands_to_check = [brand_name.lower()]
        
        if include_typos:
            typos = list(TyposquatGenerator.generate_variants(brand_name))[:3]
            brands_to_check.extend(typos)
        
        for brand in brands_to_check:
            for template in cls.GITHUB_CODE_TEMPLATES:
                query = template.format(brand=brand, domain=brand_domain)
                queries.append(query)
        
        return queries
    
    @classmethod
    def generate_github_repo_queries(cls, brand_name: str, include_typos: bool = True) -> List[str]:
        """Generate GitHub repository search queries"""
        queries = []
        brands_to_check = [brand_name.lower()]
        
        if include_typos:
            typos = list(TyposquatGenerator.generate_variants(brand_name))[:3]
            brands_to_check.extend(typos)
        
        for brand in brands_to_check:
            for template in cls.GITHUB_REPO_TEMPLATES:
                queries.append(template.format(brand=brand))
        
        return queries
    
    @classmethod
    def generate_telegram_usernames(cls, brand_name: str, include_typos: bool = True) -> List[str]:
        """Generate Telegram username patterns to check"""
        usernames = []
        brands_to_check = [brand_name.lower()]
        
        if include_typos:
            typos = list(TyposquatGenerator.generate_variants(brand_name))[:5]
            brands_to_check.extend(typos)
        
        for brand in brands_to_check:
            for template in cls.TELEGRAM_USERNAME_TEMPLATES:
                username = template.format(brand=brand)
                # Clean for Telegram (5-32 chars, alphanumeric + underscore)
                username = re.sub(r'[^a-zA-Z0-9_]', '', username)
                if 5 <= len(username) <= 32:
                    usernames.append(username)
        
        return list(set(usernames))  # Remove duplicates


class SocialMediaMonitor:
    """Monitor social media platforms for brand protection"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        """
        Initialize the social media monitor
        
        Args:
            api_keys: Dictionary containing API keys:
                - youtube: YouTube Data API v3 key
                - telegram: Telegram Bot token
                - github: GitHub personal access token
        """
        self.api_keys = api_keys or {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BrandProtection/1.0'
        })
        
    def monitor_brand(self, brand_domain: str, brand_name: str = None) -> Dict[str, Any]:
        """
        Monitor all social media platforms for brand mentions
        
        Args:
            brand_domain: The brand's domain (e.g., 'instagram.com')
            brand_name: Optional brand name (derived from domain if not provided)
        
        Returns:
            Dictionary containing monitoring results and risk assessment
        """
        # Extract brand name from domain if not provided
        if not brand_name:
            brand_name = brand_domain.split('.')[0].lower()
        
        print(f"[Social Media] Starting monitoring for brand: {brand_name} ({brand_domain})")
        
        results = {
            'brand_domain': brand_domain,
            'brand_name': brand_name,
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'threats_found': [],
            'summary': {
                'total_sources': 0,
                'sources_with_findings': 0,
                'total_threats': 0,
                'high_risk_indicators': 0
            }
        }
        
        # Check each platform
        print(f"[Social Media] Checking YouTube...")
        results['sources']['youtube'] = self.check_youtube(brand_name)
        
        print(f"[Social Media] Checking Telegram...")
        results['sources']['telegram'] = self.check_telegram(brand_name)
        
        print(f"[Social Media] Checking GitHub...")
        results['sources']['github'] = self.check_github(brand_name, brand_domain)
        
        # Calculate summary
        for source_name, source_data in results['sources'].items():
            results['summary']['total_sources'] += 1
            if source_data.get('has_findings', False):
                results['summary']['sources_with_findings'] += 1
            
            # Add threats from each source
            if source_data.get('threats'):
                results['threats_found'].extend(source_data['threats'])
                results['summary']['total_threats'] += len(source_data['threats'])
            
            # Count high risk indicators
            if source_data.get('high_risk_count', 0) > 0:
                results['summary']['high_risk_indicators'] += source_data['high_risk_count']
        
        # Calculate risk score
        results['risk_score'] = self.calculate_risk_score(results)
        results['risk_level'] = self._get_risk_level(results['risk_score'])
        
        print(f"[Social Media] Monitoring complete. Risk Score: {results['risk_score']}/100")
        return results
    
    def check_youtube(self, brand_name: str) -> Dict[str, Any]:
        """
        Check YouTube for REAL THREATS: phishing, scams, credential theft, impersonation
        
        Uses dynamic threat keyword templates and typosquatting variants.
        
        Args:
            brand_name: Brand name to search for
        
        Returns:
            Dictionary with YouTube findings
        """
        result = {
            'source': 'YouTube',
            'status': 'pending',
            'has_findings': False,
            'threats': [],
            'high_risk_count': 0,
            'channels_found': [],
            'videos_found': [],
            'all_videos_scanned': [],  # All videos for display
            'total_channels': 0,
            'total_videos': 0,
            'suspicious_channels': 0,
            'suspicious_videos': 0,
            'queries_used': [],
            'typosquats_checked': []
        }
        
        api_key = self.api_keys.get('youtube', '').strip()
        if not api_key:
            result['status'] = 'no_api_key'
            result['message'] = 'YouTube API key not provided'
            return result
        
        try:
            # Generate threat-specific queries with typosquatting variants
            threat_queries = ThreatKeywordGenerator.generate_youtube_queries(brand_name, include_typos=True)
            
            # Limit queries to avoid API quota exhaustion (select most important ones)
            # Prioritize: hack, phishing, scam, bypass queries
            priority_keywords = ['hack', 'phishing', 'phish', 'scam', 'bypass', 'steal', 'fake']
            priority_queries = [q for q in threat_queries if any(k in q.lower() for k in priority_keywords)]
            other_queries = [q for q in threat_queries if q not in priority_queries]
            
            # Take top 15 priority + 5 others
            selected_queries = priority_queries[:15] + other_queries[:5]
            result['queries_used'] = selected_queries[:10]  # Store sample for report
            
            # Track typosquats used
            typos = list(TyposquatGenerator.generate_variants(brand_name))[:5]
            result['typosquats_checked'] = typos
            
            all_videos = []
            
            for query in selected_queries:
                videos = self._youtube_search(api_key, query, 'video', max_results=3)
                all_videos.extend(videos)
                time.sleep(0.1)  # Rate limiting
            
            # Remove duplicates
            seen_video_ids = set()
            unique_videos = []
            for video in all_videos:
                if video['id'] not in seen_video_ids:
                    seen_video_ids.add(video['id'])
                    unique_videos.append(video)
            
            # STRICT threat detection patterns - brand agnostic
            threat_patterns = [
                # Credential theft / hacking
                (r'\b(hack|hacking|steal|crack|breach)\b.*\b(account|password|login|credential|email)', 'Credential theft/hacking'),
                (r'\b(password|account|login)\b.*\b(hack|steal|crack|breach|dump)', 'Credential theft/hacking'),
                
                # Phishing
                (r'\b(phishing|phish)\b', 'Phishing content'),
                (r'\b(fake|clone|replica)\b.*\b(login|page|site|website|portal)', 'Fake login page'),
                (r'\bcreate\b.*\b(fake|phishing)\b.*\b(page|site|login)', 'Phishing tutorial'),
                
                # Scams
                (r'\b(free)\b.*\b(money|cash|dollars?|gift\s*card|premium|coins?)', 'Financial scam'),
                (r'\b(unlimited)\b.*\b(free|money|coins?|credits?|premium)', 'Unlimited free scam'),
                (r'\bgenerator\b.*\b(account|password|code|gift|token|key)', 'Fake generator'),
                (r'\b(get|earn)\b.*\b(free|unlimited)\b.*\b(money|cash|coins?)', 'Money scam'),
                
                # Security bypass
                (r'\b(bypass|skip|disable)\b.*\b(verification|security|2fa|otp|captcha)', 'Security bypass'),
                (r'\b(2fa|otp|verification)\b.*\b(bypass|skip|hack|crack)', 'Security bypass'),
                
                # Exploits / Vulnerabilities
                (r'\b(exploit|vulnerability|vuln|cve)\b', 'Exploit/vulnerability'),
                (r'\b(malware|trojan|keylogger|rat|backdoor)\b', 'Malware content'),
                
                # Impersonation
                (r'\b(impersonate|pretend|pose)\b.*\b(support|admin|official|staff)', 'Impersonation tutorial'),
            ]
            
            # Analyze all videos
            for video in unique_videos:
                title_lower = video['title'].lower()
                desc_lower = (video.get('description', '') or '').lower()
                combined_text = f"{title_lower} {desc_lower}"
                
                is_threat = False
                threat_reasons = []
                threat_type = None
                
                for pattern, reason in threat_patterns:
                    if re.search(pattern, combined_text, re.IGNORECASE):
                        is_threat = True
                        threat_reasons.append(reason)
                        threat_type = reason
                        break
                
                video['is_suspicious'] = is_threat
                video['threat_type'] = threat_type
                video['suspicion_reasons'] = threat_reasons
                
                if is_threat:
                    result['suspicious_videos'] += 1
                    result['high_risk_count'] += 1
                    result['threats'].append(f"YouTube: {threat_type} - '{video['title'][:50]}...'")
            
            # Store all videos for full display
            result['all_videos_scanned'] = unique_videos
            result['videos_found'] = [v for v in unique_videos if v.get('is_suspicious')]
            result['total_videos'] = len(unique_videos)
            result['has_findings'] = result['suspicious_videos'] > 0
            result['status'] = 'success'
            result['summary'] = f"Scanned {result['total_videos']} videos using {len(selected_queries)} threat queries (including typosquats). Found {result['suspicious_videos']} potential threats."
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()
        
        return result
    
    def _youtube_search(self, api_key: str, query: str, search_type: str, max_results: int = 10) -> List[Dict]:
        """
        Perform a YouTube API search
        
        Args:
            api_key: YouTube API key
            query: Search query
            search_type: 'channel' or 'video'
            max_results: Maximum number of results
        
        Returns:
            List of search results
        """
        results = []
        
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            'key': api_key,
            'q': query,
            'type': search_type,
            'part': 'snippet',
            'maxResults': max_results,
            'order': 'relevance'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                for item in data.get('items', []):
                    snippet = item.get('snippet', {})
                    
                    if search_type == 'channel':
                        results.append({
                            'id': item.get('id', {}).get('channelId', ''),
                            'title': snippet.get('channelTitle', snippet.get('title', '')),
                            'description': snippet.get('description', ''),
                            'thumbnail': snippet.get('thumbnails', {}).get('default', {}).get('url', ''),
                            'url': f"https://www.youtube.com/channel/{item.get('id', {}).get('channelId', '')}"
                        })
                    else:  # video
                        results.append({
                            'id': item.get('id', {}).get('videoId', ''),
                            'title': snippet.get('title', ''),
                            'description': snippet.get('description', ''),
                            'channel': snippet.get('channelTitle', ''),
                            'thumbnail': snippet.get('thumbnails', {}).get('default', {}).get('url', ''),
                            'url': f"https://www.youtube.com/watch?v={item.get('id', {}).get('videoId', '')}",
                            'published_at': snippet.get('publishedAt', '')
                        })
            elif response.status_code == 403:
                print(f"[YouTube] API quota exceeded or access denied")
            else:
                print(f"[YouTube] API error: {response.status_code}")
                
        except Exception as e:
            print(f"[YouTube] Search error: {e}")
        
        return results
    
    def check_telegram(self, brand_name: str) -> Dict[str, Any]:
        """
        Check Telegram for SCAM/FRAUD impersonation using Bot API and web checks
        
        Uses dynamic username generation with typosquatting variants.
        
        Args:
            brand_name: Brand name to check
        
        Returns:
            Dictionary with Telegram findings
        """
        result = {
            'source': 'Telegram',
            'status': 'pending',
            'has_findings': False,
            'threats': [],
            'high_risk_count': 0,
            'usernames_checked': [],
            'taken_usernames': [],
            'impersonation_risk': [],
            'typosquats_checked': [],
            'scam_categories': {
                'fake_support': [],
                'giveaway_scam': [],
                'recovery_scam': [],
                'verification_scam': [],
                'admin_impersonation': [],
                'other': []
            }
        }
        
        bot_token = self.api_keys.get('telegram', '').strip()
        
        try:
            # Generate scam username patterns with typosquatting
            scam_usernames = ThreatKeywordGenerator.generate_telegram_usernames(brand_name, include_typos=True)
            
            # Track typosquats
            typos = list(TyposquatGenerator.generate_variants(brand_name))[:5]
            result['typosquats_checked'] = typos
            
            # Limit to reasonable number
            scam_usernames = scam_usernames[:50]
            
            for username in scam_usernames:
                # Try Bot API first if token available, then fallback to web
                if bot_token:
                    is_taken = self._check_telegram_via_bot(bot_token, username)
                else:
                    is_taken = self._check_telegram_username(username)
                
                check_result = {
                    'username': f"@{username}",
                    'is_taken': is_taken,
                    'url': f"https://t.me/{username}"
                }
                
                result['usernames_checked'].append(check_result)
                
                if is_taken:
                    result['taken_usernames'].append(username)
                    
                    # Categorize by scam type
                    scam_type, scam_category = self._categorize_telegram_scam(username)
                    
                    scam_entry = {
                        'username': f"@{username}",
                        'reason': scam_type,
                        'category': scam_category,
                        'url': f"https://t.me/{username}"
                    }
                    
                    result['impersonation_risk'].append(scam_entry)
                    result['scam_categories'][scam_category].append(scam_entry)
                    result['high_risk_count'] += 1
                    result['threats'].append(f"Telegram: @{username} - {scam_type}")
                
                # Rate limiting
                time.sleep(0.2)
            
            result['has_findings'] = len(result['impersonation_risk']) > 0
            result['status'] = 'success'
            
            # Generate detailed summary
            if result['has_findings']:
                category_counts = {k: len(v) for k, v in result['scam_categories'].items() if v}
                category_str = ', '.join([f"{k.replace('_', ' ').title()}: {v}" for k, v in category_counts.items()])
                result['summary'] = f"⚠️ ALERT: Found {len(result['impersonation_risk'])} potential SCAM accounts. Breakdown: {category_str}. Checked {len(result['usernames_checked'])} patterns including typosquats."
            else:
                result['summary'] = f"Checked {len(result['usernames_checked'])} high-risk username patterns (including typosquats of '{brand_name}'). No active scam accounts found."
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()
        
        return result
    
    def _check_telegram_via_bot(self, bot_token: str, username: str) -> bool:
        """Check if a Telegram username exists using Bot API"""
        try:
            # Try to get chat info via Bot API
            url = f"https://api.telegram.org/bot{bot_token}/getChat"
            params = {'chat_id': f"@{username}"}
            
            response = self.session.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get('ok') and data.get('result'):
                return True
            
            # If bot API fails, fallback to web check
            return self._check_telegram_username(username)
            
        except Exception:
            # Fallback to web check
            return self._check_telegram_username(username)
    
    def _categorize_telegram_scam(self, username: str) -> tuple:
        """Categorize the type of Telegram scam based on username"""
        username_lower = username.lower()
        
        # Fake support
        if any(kw in username_lower for kw in ['support', 'helpdesk', 'customer', 'service', 'help']):
            return ("⚠️ FAKE SUPPORT - Tricks users into giving credentials/money", 'fake_support')
        
        # Giveaway scams
        if any(kw in username_lower for kw in ['giveaway', 'airdrop', 'free', 'prize', 'reward']):
            return ("⚠️ GIVEAWAY SCAM - Promotes fake giveaways to steal money", 'giveaway_scam')
        
        # Recovery scams
        if any(kw in username_lower for kw in ['recovery', 'restore', 'unlock', 'recover']):
            return ("⚠️ RECOVERY SCAM - Targets hacked users to steal more", 'recovery_scam')
        
        # Verification scams
        if any(kw in username_lower for kw in ['verify', 'verification', 'verified']):
            return ("⚠️ VERIFICATION SCAM - Fake verification to steal credentials", 'verification_scam')
        
        # Admin impersonation
        if any(kw in username_lower for kw in ['admin', 'mod', 'moderator', 'team', 'staff', 'official']):
            return ("⚠️ ADMIN IMPERSONATION - Pretends to be platform staff", 'admin_impersonation')
        
        return ("⚠️ POTENTIAL IMPERSONATION", 'other')
    
    def _check_telegram_username(self, username: str) -> bool:
        """
        Check if a Telegram username is taken
        
        Args:
            username: Username to check (without @)
        
        Returns:
            True if username is taken, False if available
        """
        try:
            url = f"https://t.me/{username}"
            response = self.session.head(url, timeout=10, allow_redirects=True)
            
            # If we get redirected to a preview page or get 200, username exists
            # If we get a redirect to t.me/username where username doesn't exist, we get different response
            if response.status_code == 200:
                # Check if page contains "tgme_page_photo" (indicates real account)
                try:
                    response_get = self.session.get(url, timeout=10)
                    if 'tgme_page_photo' in response_get.text or 'tgme_page_title' in response_get.text:
                        return True
                    # If page says "You can contact" or shows a preview, account exists
                    if 'You can contact' in response_get.text or 'tgme_page_description' in response_get.text:
                        return True
                except Exception:
                    return True  # Assume taken if we can't verify
            
            return False
            
        except Exception as e:
            print(f"[Telegram] Error checking @{username}: {e}")
            return False
    
    def check_github(self, brand_name: str, brand_domain: str) -> Dict[str, Any]:
        """
        Check GitHub for exposed secrets, phishing kits, and leaked code
        
        Uses dynamic keyword generation with typosquatting variants.
        
        Args:
            brand_name: Brand name to search for
            brand_domain: Brand domain for email pattern search
        
        Returns:
            Dictionary with GitHub findings
        """
        result = {
            'source': 'GitHub',
            'status': 'pending',
            'has_findings': False,
            'threats': [],
            'high_risk_count': 0,
            'code_results': [],
            'repo_results': [],
            'all_code_scanned': [],
            'all_repos_scanned': [],
            'exposed_secrets': [],
            'phishing_kits': [],
            'credential_leaks': [],
            'total_code_matches': 0,
            'total_repos': 0,
            'queries_used': [],
            'typosquats_checked': []
        }
        
        github_token = self.api_keys.get('github', '').strip()
        
        headers = {
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28'
        }
        
        if github_token:
            headers['Authorization'] = f'Bearer {github_token}'
        
        try:
            # Generate code search queries with typosquatting
            code_queries = ThreatKeywordGenerator.generate_github_code_queries(brand_name, brand_domain, include_typos=True)
            repo_queries = ThreatKeywordGenerator.generate_github_repo_queries(brand_name, include_typos=True)
            
            # Track typosquats
            typos = list(TyposquatGenerator.generate_variants(brand_name))[:3]
            result['typosquats_checked'] = typos
            
            # Limit queries to avoid rate limiting
            code_queries = code_queries[:15]
            repo_queries = repo_queries[:10]
            
            result['queries_used'] = code_queries[:5] + repo_queries[:3]
            
            # Code search
            for query in code_queries:
                code_results = self._github_search_code(query, headers)
                
                for item in code_results:
                    # Categorize the finding
                    threat_type = self._categorize_github_threat(query, item)
                    item['threat_type'] = threat_type
                    item['query'] = query
                    
                    result['all_code_scanned'].append(item)
                    result['total_code_matches'] += 1
                    
                    if threat_type == 'exposed_secret':
                        result['exposed_secrets'].append(item)
                        result['high_risk_count'] += 1
                        result['threats'].append(f"GitHub: Exposed secret in {item['repo']}/{item['path']}")
                    elif threat_type == 'phishing_kit':
                        result['phishing_kits'].append(item)
                        result['high_risk_count'] += 1
                        result['threats'].append(f"GitHub: Phishing kit code in {item['repo']}")
                    elif threat_type == 'credential_leak':
                        result['credential_leaks'].append(item)
                        result['high_risk_count'] += 1
                        result['threats'].append(f"GitHub: Credential leak in {item['repo']}")
                
                time.sleep(0.5)  # Rate limiting
            
            # Repository search
            for query in repo_queries:
                repo_results = self._github_search_repos(query, headers)
                
                for item in repo_results:
                    result['all_repos_scanned'].append(item)
                    result['total_repos'] += 1
                    
                    # Check for threat indicators
                    name_lower = item['name'].lower()
                    desc_lower = (item.get('description') or '').lower()
                    combined = f"{name_lower} {desc_lower}"
                    
                    threat_keywords = [
                        ('phishing', 'Phishing kit repository'),
                        ('phish', 'Phishing kit repository'),
                        ('fake login', 'Fake login page'),
                        ('credential', 'Credential harvester'),
                        ('scam', 'Scam-related repository'),
                        ('hack', 'Hacking tool'),
                        ('exploit', 'Exploit code'),
                        ('stealer', 'Credential stealer'),
                    ]
                    
                    for keyword, description in threat_keywords:
                        if keyword in combined:
                            item['threat_type'] = description
                            item['is_suspicious'] = True
                            result['phishing_kits'].append(item)
                            result['high_risk_count'] += 1
                            result['threats'].append(f"GitHub: {description} - '{item['name']}'")
                            break
                    else:
                        item['is_suspicious'] = False
                
                time.sleep(0.5)
            
            result['code_results'] = result['exposed_secrets'] + result['phishing_kits'] + result['credential_leaks']
            result['repo_results'] = [r for r in result['all_repos_scanned'] if r.get('is_suspicious')]
            result['has_findings'] = len(result['threats']) > 0
            result['status'] = 'success'
            result['summary'] = f"Scanned {result['total_code_matches']} code files, {result['total_repos']} repositories (including typosquats). Found: {len(result['exposed_secrets'])} exposed secrets, {len(result['phishing_kits'])} phishing kits, {len(result['credential_leaks'])} credential leaks."
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()
        
        return result
    
    def _categorize_github_threat(self, query: str, item: Dict) -> str:
        """Categorize the type of GitHub threat based on query and content"""
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ['api_key', 'secret', 'token', 'password', 'credential']):
            return 'exposed_secret'
        elif any(kw in query_lower for kw in ['phishing', 'phish', 'fake', 'login']):
            return 'phishing_kit'
        elif '@' in query_lower:
            return 'credential_leak'
        else:
            return 'other'
    
    def _github_search_code(self, query: str, headers: Dict) -> List[Dict]:
        """Search GitHub code"""
        results = []
        
        try:
            url = "https://api.github.com/search/code"
            params = {
                'q': query,
                'per_page': 10
            }
            
            response = self.session.get(url, headers=headers, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                for item in data.get('items', []):
                    results.append({
                        'name': item.get('name', ''),
                        'path': item.get('path', ''),
                        'repo': item.get('repository', {}).get('full_name', ''),
                        'url': item.get('html_url', ''),
                        'score': item.get('score', 0)
                    })
            elif response.status_code == 403:
                print(f"[GitHub] Rate limited or access denied")
            elif response.status_code == 422:
                # Query too complex or no results
                pass
            else:
                print(f"[GitHub] Code search error: {response.status_code}")
                
        except Exception as e:
            print(f"[GitHub] Code search error: {e}")
        
        return results
    
    def _github_search_repos(self, query: str, headers: Dict) -> List[Dict]:
        """Search GitHub repositories"""
        results = []
        
        try:
            url = "https://api.github.com/search/repositories"
            params = {
                'q': query,
                'per_page': 10,
                'sort': 'updated'
            }
            
            response = self.session.get(url, headers=headers, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                for item in data.get('items', []):
                    results.append({
                        'name': item.get('name', ''),
                        'full_name': item.get('full_name', ''),
                        'description': item.get('description', ''),
                        'url': item.get('html_url', ''),
                        'stars': item.get('stargazers_count', 0),
                        'forks': item.get('forks_count', 0),
                        'created_at': item.get('created_at', ''),
                        'updated_at': item.get('updated_at', '')
                    })
            elif response.status_code == 403:
                print(f"[GitHub] Rate limited or access denied")
            else:
                print(f"[GitHub] Repo search error: {response.status_code}")
                
        except Exception as e:
            print(f"[GitHub] Repo search error: {e}")
        
        return results
    
    def calculate_risk_score(self, results: Dict) -> int:
        """
        Calculate overall risk score based on findings
        
        Args:
            results: Monitoring results dictionary
        
        Returns:
            Risk score from 0-100
        """
        score = 0
        
        # YouTube findings
        youtube = results['sources'].get('youtube', {})
        if youtube.get('status') == 'success':
            score += min(youtube.get('suspicious_channels', 0) * 10, 30)
            score += min(youtube.get('suspicious_videos', 0) * 5, 20)
        
        # Telegram findings
        telegram = results['sources'].get('telegram', {})
        if telegram.get('status') == 'success':
            score += min(len(telegram.get('impersonation_risk', [])) * 15, 30)
        
        # GitHub findings
        github = results['sources'].get('github', {})
        if github.get('status') == 'success':
            score += min(len(github.get('exposed_secrets', [])) * 20, 40)
            score += min(len(github.get('phishing_kits', [])) * 15, 30)
        
        return min(score, 100)
    
    def _get_risk_level(self, score: int) -> str:
        """Get risk level label from score"""
        if score >= 70:
            return "Critical"
        elif score >= 50:
            return "High"
        elif score >= 30:
            return "Medium"
        elif score >= 10:
            return "Low"
        else:
            return "Minimal"
    
    def generate_report(self, results: Dict, output_path: Path = None) -> str:
        """
        Generate HTML report from monitoring results
        
        Args:
            results: Monitoring results dictionary
            output_path: Optional path to save report
        
        Returns:
            HTML report string
        """
        risk_score = results.get('risk_score', 0)
        risk_level = results.get('risk_level', 'Unknown')
        
        # Determine risk color
        if risk_score >= 70:
            risk_color = '#dc3545'
            risk_bg = '#f8d7da'
        elif risk_score >= 50:
            risk_color = '#fd7e14'
            risk_bg = '#fff3cd'
        elif risk_score >= 30:
            risk_color = '#ffc107'
            risk_bg = '#fff9e6'
        elif risk_score >= 10:
            risk_color = '#0dcaf0'
            risk_bg = '#d1ecf1'
        else:
            risk_color = '#198754'
            risk_bg = '#d1e7dd'
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Social Media Monitoring Report - {results.get('brand_name', 'Unknown')}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css" rel="stylesheet">
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f8f9fa; }}
        .report-header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; margin-bottom: 30px; }}
        .risk-score {{ font-size: 72px; font-weight: bold; color: {risk_color}; }}
        .risk-badge {{ background: {risk_bg}; color: {risk_color}; padding: 10px 20px; border-radius: 25px; font-weight: bold; }}
        .source-card {{ border-left: 4px solid #667eea; margin-bottom: 20px; }}
        .threat-item {{ border-left: 3px solid #dc3545; padding-left: 15px; margin: 10px 0; }}
        .finding-card {{ background: white; border-radius: 8px; padding: 15px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <div class="report-header">
        <div class="container">
            <h1><i class="bi bi-share-fill"></i> Social Media Monitoring Report</h1>
            <p class="lead mb-0">Brand: <strong>{results.get('brand_name', 'Unknown')}</strong> ({results.get('brand_domain', 'Unknown')})</p>
            <p class="mb-0">Generated: {results.get('timestamp', datetime.now().isoformat())}</p>
        </div>
    </div>
    
    <div class="container">
        <!-- Risk Score Section -->
        <div class="row mb-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-body text-center">
                        <h3>Overall Risk Assessment</h3>
                        <div class="risk-score">{risk_score}/100</div>
                        <span class="risk-badge">{risk_level} Risk</span>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Summary Cards -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card bg-primary text-white">
                    <div class="card-body text-center">
                        <h3>{results['summary']['total_sources']}</h3>
                        <p class="mb-0">Sources Checked</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-danger text-white">
                    <div class="card-body text-center">
                        <h3>{results['summary']['sources_with_findings']}</h3>
                        <p class="mb-0">Sources with Findings</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-warning text-white">
                    <div class="card-body text-center">
                        <h3>{results['summary']['high_risk_indicators']}</h3>
                        <p class="mb-0">High Risk Indicators</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-info text-white">
                    <div class="card-body text-center">
                        <h3>{results['summary']['total_threats']}</h3>
                        <p class="mb-0">Total Threats</p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Threats Found -->
        <div class="card mb-4">
            <div class="card-header bg-danger text-white">
                <h5 class="mb-0"><i class="bi bi-exclamation-triangle-fill"></i> Threats Identified</h5>
            </div>
            <div class="card-body">
                {"".join(f'<div class="threat-item">{t}</div>' for t in results.get('threats_found', [])) if results.get('threats_found') else '<p class="text-success"><i class="bi bi-check-circle"></i> No threats identified.</p>'}
            </div>
        </div>
"""
        
        # Add source-specific sections
        sources = results.get('sources', {})
        
        # YouTube Section
        youtube = sources.get('youtube', {})
        if youtube.get('status') == 'success':
            html += f"""
        <div class="card source-card mb-4">
            <div class="card-header">
                <h5 class="mb-0"><i class="bi bi-youtube text-danger"></i> YouTube Findings</h5>
            </div>
            <div class="card-body">
                <p>{youtube.get('summary', '')}</p>
                <div class="row">
                    <div class="col-md-6">
                        <h6>Suspicious Channels ({youtube.get('suspicious_channels', 0)})</h6>
                        {"".join(f'''<div class="finding-card">
                            <strong><a href="{c['url']}" target="_blank">{c['title']}</a></strong>
                            {f"<br><small class='text-danger'>{', '.join(c.get('suspicion_reasons', []))}</small>" if c.get('is_suspicious') else ""}
                        </div>''' for c in youtube.get('channels_found', []) if c.get('is_suspicious')) or '<p class="text-muted">None found</p>'}
                    </div>
                    <div class="col-md-6">
                        <h6>Suspicious Videos ({youtube.get('suspicious_videos', 0)})</h6>
                        {"".join(f'''<div class="finding-card">
                            <strong><a href="{v['url']}" target="_blank">{v['title'][:60]}...</a></strong>
                            <br><small>Channel: {v.get('channel', 'Unknown')}</small>
                        </div>''' for v in youtube.get('videos_found', []) if v.get('is_suspicious')) or '<p class="text-muted">None found</p>'}
                    </div>
                </div>
            </div>
        </div>
"""
        
        # Telegram Section
        telegram = sources.get('telegram', {})
        if telegram.get('status') == 'success':
            html += f"""
        <div class="card source-card mb-4">
            <div class="card-header">
                <h5 class="mb-0"><i class="bi bi-telegram text-primary"></i> Telegram Findings</h5>
            </div>
            <div class="card-body">
                <p>{telegram.get('summary', '')}</p>
                <h6>Potential Impersonation Accounts ({len(telegram.get('impersonation_risk', []))})</h6>
                {"".join(f'''<div class="finding-card">
                    <strong><a href="{imp['url']}" target="_blank">{imp['username']}</a></strong>
                    <br><small class="text-danger">{imp['reason']}</small>
                </div>''' for imp in telegram.get('impersonation_risk', [])) or '<p class="text-muted">No high-risk accounts found</p>'}
                
                <h6 class="mt-3">All Taken Usernames</h6>
                <p>{"".join(f'<span class="badge bg-secondary me-1">@{u}</span>' for u in telegram.get('taken_usernames', [])) or '<span class="text-muted">None</span>'}</p>
            </div>
        </div>
"""
        
        # GitHub Section
        github = sources.get('github', {})
        if github.get('status') == 'success':
            html += f"""
        <div class="card source-card mb-4">
            <div class="card-header">
                <h5 class="mb-0"><i class="bi bi-github"></i> GitHub Findings</h5>
            </div>
            <div class="card-body">
                <p>{github.get('summary', '')}</p>
                <div class="row">
                    <div class="col-md-6">
                        <h6>Exposed Secrets ({len(github.get('exposed_secrets', []))})</h6>
                        {"".join(f'''<div class="finding-card">
                            <strong><a href="{s['url']}" target="_blank">{s['repo']}</a></strong>
                            <br><small>File: {s['path']}</small>
                        </div>''' for s in github.get('exposed_secrets', [])) or '<p class="text-success">No exposed secrets found</p>'}
                    </div>
                    <div class="col-md-6">
                        <h6>Potential Phishing Kits ({len(github.get('phishing_kits', []))})</h6>
                        {"".join(f'''<div class="finding-card">
                            <strong><a href="{p.get('url', '#')}" target="_blank">{p.get('name', p.get('repo', 'Unknown'))}</a></strong>
                            {f"<br><small>{p.get('description', '')[:100]}</small>" if p.get('description') else ""}
                        </div>''' for p in github.get('phishing_kits', [])) or '<p class="text-success">No phishing kits found</p>'}
                    </div>
                </div>
            </div>
        </div>
"""
        
        html += """
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
        
        return html


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Social Media Monitoring for Brand Protection')
    parser.add_argument('domain', help='Brand domain to monitor (e.g., instagram.com)')
    parser.add_argument('--brand-name', help='Brand name (derived from domain if not provided)')
    parser.add_argument('--output', '-o', help='Output HTML report file')
    
    args = parser.parse_args()
    
    # Load API keys from environment
    api_keys = {
        'youtube': os.getenv('YOUTUBE_API_KEY', ''),
        'telegram': os.getenv('TELEGRAM_BOT_TOKEN', ''),
        'github': os.getenv('GITHUB_TOKEN', '')
    }
    
    monitor = SocialMediaMonitor(api_keys=api_keys)
    results = monitor.monitor_brand(args.domain, args.brand_name)
    
    # Print summary
    print("\n" + "="*60)
    print(f"SOCIAL MEDIA MONITORING REPORT - {results['brand_name']}")
    print("="*60)
    print(f"Risk Score: {results['risk_score']}/100 ({results['risk_level']})")
    print(f"Sources Checked: {results['summary']['total_sources']}")
    print(f"Sources with Findings: {results['summary']['sources_with_findings']}")
    print(f"Total Threats: {results['summary']['total_threats']}")
    print("\nThreats Found:")
    for threat in results.get('threats_found', []):
        print(f"  - {threat}")
    
    # Generate report if output specified
    if args.output:
        monitor.generate_report(results, Path(args.output))
        print(f"\nReport saved to: {args.output}")

