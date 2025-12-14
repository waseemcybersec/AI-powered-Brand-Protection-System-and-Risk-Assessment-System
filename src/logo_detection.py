#!/usr/bin/env python3
"""
Advanced Logo Detection Script - Multi-Method Ensemble Approach

This script detects brand logos in domain screenshots using a comprehensive ensemble approach:
1. Multi-Scale Template Matching (OpenCV) - For exact/near-exact logo matches at different scales
2. DINOv2 Vision Transformer - For semantic similarity and logo variations
3. Perceptual Hashing (pHash) - For quick filtering and validation
4. Ensemble Voting - Combines all methods with weighted confidence scoring
5. Adaptive Thresholding - Uses relative thresholds (match gap) instead of fixed thresholds

This approach significantly reduces both false positives and false negatives by:
- Using multiple independent detection methods
- Requiring consensus or strong evidence from at least one method
- Using adaptive thresholds based on match quality (gap between best and second-best)
- Multi-scale detection to catch logos at different sizes

Usage:
    python logo_detection.py [--input dataset_final.csv] [--output dataset_with_logo.csv]
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
from PIL import Image
import imagehash
import cv2

# DINOv2 imports
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, AutoModel
    DINO_AVAILABLE = True
except ImportError:
    DINO_AVAILABLE = False
    print("[ERROR] DINOv2 libraries are required for logo detection.")
    print("[INFO] Install with: pip install torch transformers")
    sys.exit(1)


class LogoDetector:
    """Advanced logo detector using multi-method ensemble approach."""
    
    def __init__(self, reference_logo_paths, similarity_threshold=0.88, phash_threshold=2, template_threshold=0.88):
        """
        Initialize the logo detector with EXTREMELY STRICT thresholds to eliminate false positives.
        
        Args:
            reference_logo_paths: List of paths to reference logo images
            similarity_threshold: Base DINOv2 similarity threshold (EXTREMELY STRICT: 0.88 = 88%)
            phash_threshold: Maximum pHash distance (EXTREMELY STRICT: 2, very conservative)
            template_threshold: Template matching threshold (EXTREMELY STRICT: 0.88 = 88%)
        """
        # Handle single path or list of paths
        if isinstance(reference_logo_paths, (str, Path)):
            reference_logo_paths = [reference_logo_paths]
        
        self.reference_logo_paths = [Path(p) for p in reference_logo_paths]
        self.base_similarity_threshold = similarity_threshold
        self.base_phash_threshold = phash_threshold
        self.template_threshold = template_threshold
        
        # Load all reference logos
        self.reference_logos = []
        self.reference_phashes = []
        self.reference_templates = []  # OpenCV templates for template matching
        
        print("[INFO] Loading reference logos...")
        for ref_path in self.reference_logo_paths:
            if not ref_path.exists():
                print(f"[WARNING] Reference logo not found: {ref_path}, skipping...")
                continue
            
            print(f"[INFO] Processing reference logo: {ref_path}")
            try:
                # Load image with PIL
                ref_image = Image.open(ref_path).convert('RGB')
                self.reference_logos.append(ref_image)
                
                # Compute pHash
                ref_phash = imagehash.phash(ref_image)
                self.reference_phashes.append(ref_phash)
                
                # Convert to OpenCV format for template matching
                ref_cv = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
                self.reference_templates.append(ref_cv)
                
                print(f"[INFO]   - Loaded: {ref_image.size[0]}x{ref_image.size[1]}, pHash: {ref_phash}")
                
            except Exception as e:
                print(f"[WARNING] Failed to process reference logo {ref_path}: {e}")
                import traceback
                traceback.print_exc()
        
        if not self.reference_logos:
            raise FileNotFoundError("No valid reference logos found")
        
        print(f"[INFO] Loaded {len(self.reference_logos)} reference logo(s)")
        
        # Initialize DINOv2 model
        self.processor = None
        self.model = None
        self.reference_embeddings = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if DINO_AVAILABLE:
            try:
                print("[INFO] Loading DINOv2 model (this may take a moment)...")
                print(f"[INFO] Using device: {self.device}")
                
                model_name = "facebook/dinov2-base"
                self.processor = AutoImageProcessor.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name).to(self.device)
                self.model.eval()
                
                # Compute reference DINOv2 embeddings
                print("[INFO] Computing reference DINOv2 embeddings...")
                with torch.no_grad():
                    for ref_image in self.reference_logos:
                        inputs = self.processor(images=ref_image, return_tensors="pt")
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        outputs = self.model(**inputs)
                        ref_embedding = outputs.last_hidden_state[:, 0, :]
                        ref_embedding = F.normalize(ref_embedding, p=2, dim=1)
                        self.reference_embeddings.append(ref_embedding.cpu())
                
                print(f"[INFO] DINOv2 model ready. Computed {len(self.reference_embeddings)} reference embeddings.")
            except Exception as e:
                print(f"[ERROR] Failed to load DINOv2 model: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            raise RuntimeError("DINOv2 is required for logo detection but is not available")
        
        print(f"[INFO] Logo detector initialized with EXTREMELY STRICT ensemble approach:")
        print(f"[INFO]   - Template Matching: threshold={self.template_threshold} (EXTREMELY STRICT)")
        print(f"[INFO]   - DINOv2: base threshold={self.base_similarity_threshold} (EXTREMELY STRICT, no adaptation)")
        print(f"[INFO]   - pHash: base threshold={self.base_phash_threshold} (EXTREMELY STRICT, no adaptation)")
        print(f"[INFO]   - Detection requires: ALL 3 methods MUST agree on SAME reference with high confidence (>=0.85 each)")
        print(f"[INFO]   - Margin validation: Best match must be significantly better than second-best")
        print(f"[INFO]   - NO exceptions: No single-method or 2-method detections")
    
    def extract_logo_regions(self, screenshot_path, region_sizes=[300, 200, 150]):
        """
        Extract potential logo regions from screenshot at multiple scales.
        
        Args:
            screenshot_path: Path to screenshot
            region_sizes: List of region sizes to extract (multi-scale)
        
        Returns:
            List of (region_image, region_name, bbox, scale) tuples
        """
        try:
            screenshot = Image.open(screenshot_path).convert('RGB')
            width, height = screenshot.size
            
            regions = []
            
            for region_size in region_sizes:
                if width < region_size or height < region_size:
                    continue
                
                # Top-left corner
                top_left = screenshot.crop((0, 0, region_size, region_size))
                regions.append((top_left, f'top_left_{region_size}', (0, 0, region_size, region_size), region_size))
                
                # Top-center
                center_x = max(0, (width - region_size) // 2)
                top_center = screenshot.crop((center_x, 0, center_x + region_size, region_size))
                regions.append((top_center, f'top_center_{region_size}', (center_x, 0, center_x + region_size, region_size), region_size))
                
                # Top-right corner
                top_right_x = max(0, width - region_size)
                top_right = screenshot.crop((top_right_x, 0, width, region_size))
                regions.append((top_right, f'top_right_{region_size}', (top_right_x, 0, width, region_size), region_size))
                
                # Center (some logos are centered)
                center_x = max(0, (width - region_size) // 2)
                center_y = max(0, (height - region_size) // 2)
                center = screenshot.crop((center_x, center_y, center_x + region_size, center_y + region_size))
                regions.append((center, f'center_{region_size}', (center_x, center_y, center_x + region_size, center_y + region_size), region_size))
            
            return regions
        except Exception as e:
            print(f"[WARNING] Failed to extract regions: {e}")
            return []
    
    def match_template_multi_scale(self, screenshot_path):
        """
        Multi-scale template matching using OpenCV.
        Very accurate for exact/near-exact logo matches.
        
        Returns:
            Dictionary with best match results
        """
        try:
            screenshot_cv = cv2.imread(str(screenshot_path))
            if screenshot_cv is None:
                return {'matched': False, 'confidence': 0.0, 'ref_idx': -1, 'method': 'template'}
            
            screenshot_gray = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2GRAY)
            
            best_match_val = 0.0
            best_ref_idx = -1
            best_scale = 1.0
            best_location = None
            
            # Try each reference logo
            for ref_idx, template_cv in enumerate(self.reference_templates):
                template_gray = cv2.cvtColor(template_cv, cv2.COLOR_BGR2GRAY)
                template_h, template_w = template_gray.shape
                
                # Try multiple scales (logos can appear at different sizes)
                scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
                
                for scale in scales:
                    # Resize template
                    new_w = int(template_w * scale)
                    new_h = int(template_h * scale)
                    
                    if new_w > screenshot_gray.shape[1] or new_h > screenshot_gray.shape[0]:
                        continue
                    
                    resized_template = cv2.resize(template_gray, (new_w, new_h))
                    
                    # Template matching
                    result = cv2.matchTemplate(screenshot_gray, resized_template, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    
                    if max_val > best_match_val:
                        best_match_val = max_val
                        best_ref_idx = ref_idx
                        best_scale = scale
                        best_location = max_loc
            
            matched = best_match_val >= self.template_threshold
            
            return {
                'matched': matched,
                'confidence': best_match_val,
                'ref_idx': best_ref_idx,
                'method': 'template',
                'scale': best_scale,
                'location': best_location
            }
        except Exception as e:
            print(f"[WARNING] Template matching failed: {e}")
            return {'matched': False, 'confidence': 0.0, 'ref_idx': -1, 'method': 'template'}
    
    def match_logo_dinov2(self, screenshot_path):
        """
        Match logo using DINOv2 with adaptive thresholding.
        Uses region-based approach and adaptive thresholds based on match gap.
        """
        try:
            regions = self.extract_logo_regions(screenshot_path)
            
            if not regions:
                # Fallback to full image
                screenshot_image = Image.open(screenshot_path).convert('RGB')
                inputs = self.processor(images=screenshot_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    screenshot_embedding = outputs.last_hidden_state[:, 0, :]
                    screenshot_embedding = F.normalize(screenshot_embedding, p=2, dim=1)
                
                all_similarities = []
                with torch.no_grad():
                    screenshot_emb_cpu = screenshot_embedding.cpu()
                    for ref_embedding in self.reference_embeddings:
                        similarity = F.cosine_similarity(screenshot_emb_cpu, ref_embedding, dim=1).item()
                        all_similarities.append(similarity)
                
                if not all_similarities:
                    return {'matched': False, 'confidence': 0.0, 'ref_idx': -1, 'method': 'dinov2'}
                
                best_similarity = max(all_similarities)
                best_ref_idx = all_similarities.index(best_similarity)
                
                # EXTREMELY STRICT: Require large margin and high threshold
                sorted_sims = sorted(all_similarities, reverse=True)
                if len(sorted_sims) > 1:
                    gap = sorted_sims[0] - sorted_sims[1]
                    # Require minimum 0.10 gap (10%) - no threshold reduction
                    # Only proceed if gap is large enough
                    if gap < 0.10:
                        # Gap too small - reject even if above threshold
                        return {'matched': False, 'confidence': best_similarity, 'ref_idx': best_ref_idx, 'method': 'dinov2_full', 'adaptive_threshold': self.base_similarity_threshold, 'all_similarities': all_similarities, 'gap_too_small': True}
                    adaptive_threshold = self.base_similarity_threshold  # No reduction
                else:
                    adaptive_threshold = self.base_similarity_threshold
                
                matched = best_similarity >= adaptive_threshold
                
                return {
                    'matched': matched,
                    'confidence': best_similarity,
                    'ref_idx': best_ref_idx,
                    'method': 'dinov2_full',
                    'adaptive_threshold': adaptive_threshold,
                    'all_similarities': all_similarities
                }
            
            # Region-based matching
            best_similarity = 0.0
            best_ref_idx = -1
            best_region_name = None
            all_similarities = []
            
            for region_img, region_name, bbox, scale in regions:
                inputs = self.processor(images=region_img, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    region_embedding = outputs.last_hidden_state[:, 0, :]
                    region_embedding = F.normalize(region_embedding, p=2, dim=1)
                
                with torch.no_grad():
                    region_emb_cpu = region_embedding.cpu()
                    for ref_idx, ref_embedding in enumerate(self.reference_embeddings):
                        similarity = F.cosine_similarity(region_emb_cpu, ref_embedding, dim=1).item()
                        all_similarities.append(similarity)
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_ref_idx = ref_idx
                            best_region_name = region_name
            
            # EXTREMELY STRICT: Require large margin
            if all_similarities:
                sorted_sims = sorted(all_similarities, reverse=True)
                if len(sorted_sims) > 1:
                    gap = sorted_sims[0] - sorted_sims[1]
                    # Require minimum 0.10 gap (10%) - no threshold reduction
                    if gap < 0.10:
                        # Gap too small - reject
                        return {'matched': False, 'confidence': best_similarity, 'ref_idx': best_ref_idx, 'method': 'dinov2_regions', 'best_region': best_region_name, 'adaptive_threshold': self.base_similarity_threshold, 'all_similarities': all_similarities, 'gap_too_small': True}
                    adaptive_threshold = self.base_similarity_threshold  # No reduction
                else:
                    adaptive_threshold = self.base_similarity_threshold
            else:
                adaptive_threshold = self.base_similarity_threshold
            
            matched = best_similarity >= adaptive_threshold
            
            return {
                'matched': matched,
                'confidence': best_similarity,
                'ref_idx': best_ref_idx,
                'method': 'dinov2_regions',
                'best_region': best_region_name,
                'adaptive_threshold': adaptive_threshold,
                'all_similarities': all_similarities
            }
            
        except Exception as e:
            print(f"[WARNING] DINOv2 matching failed: {e}")
            return {'matched': False, 'confidence': 0.0, 'ref_idx': -1, 'method': 'dinov2'}
    
    def match_logo_phash(self, screenshot_path):
        """
        Match logo using perceptual hash with adaptive thresholding.
        """
        try:
            regions = self.extract_logo_regions(screenshot_path)
            
            if not regions:
                screenshot_image = Image.open(screenshot_path).convert('RGB')
                screenshot_phash = imagehash.phash(screenshot_image)
                
                all_distances = []
                for ref_phash in self.reference_phashes:
                    distance = screenshot_phash - ref_phash
                    all_distances.append(distance)
                
                if not all_distances:
                    return {'matched': False, 'confidence': 0.0, 'ref_idx': -1, 'method': 'phash'}
                
                best_distance = min(all_distances)
                best_ref_idx = all_distances.index(best_distance)
                confidence = max(0.0, 1.0 - (best_distance / 64.0))
                
                # EXTREMELY STRICT: Require clear margin
                sorted_dists = sorted(all_distances)
                if len(sorted_dists) > 1:
                    gap = sorted_dists[1] - sorted_dists[0]
                    # Require minimum gap of 3 units - no threshold increase
                    if gap < 3:
                        # Gap too small - reject
                        return {'matched': False, 'confidence': confidence, 'distance': best_distance, 'ref_idx': best_ref_idx, 'method': 'phash_full', 'adaptive_threshold': self.base_phash_threshold, 'all_distances': all_distances, 'gap_too_small': True}
                    adaptive_threshold = self.base_phash_threshold  # No increase
                else:
                    adaptive_threshold = self.base_phash_threshold
                
                matched = best_distance <= adaptive_threshold
                
                return {
                    'matched': matched,
                    'confidence': confidence,
                    'distance': best_distance,
                    'ref_idx': best_ref_idx,
                    'method': 'phash_full',
                    'adaptive_threshold': adaptive_threshold,
                    'all_distances': all_distances
                }
            
            # Region-based matching
            best_distance = float('inf')
            best_ref_idx = -1
            best_region_name = None
            all_distances = []
            
            for region_img, region_name, bbox, scale in regions:
                region_phash = imagehash.phash(region_img)
                
                for ref_idx, ref_phash in enumerate(self.reference_phashes):
                    distance = region_phash - ref_phash
                    all_distances.append(distance)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_ref_idx = ref_idx
                        best_region_name = region_name
            
            confidence = max(0.0, 1.0 - (best_distance / 64.0))
            
            # EXTREMELY STRICT: Require clear margin
            if all_distances:
                sorted_dists = sorted(all_distances)
                if len(sorted_dists) > 1:
                    gap = sorted_dists[1] - sorted_dists[0]
                    # Require minimum gap of 3 units - no threshold increase
                    if gap < 3:
                        # Gap too small - reject
                        return {'matched': False, 'confidence': confidence, 'distance': best_distance, 'ref_idx': best_ref_idx, 'method': 'phash_regions', 'best_region': best_region_name, 'adaptive_threshold': self.base_phash_threshold, 'all_distances': all_distances, 'gap_too_small': True}
                    adaptive_threshold = self.base_phash_threshold  # No increase
                else:
                    adaptive_threshold = self.base_phash_threshold
            else:
                adaptive_threshold = self.base_phash_threshold
            
            matched = best_distance <= adaptive_threshold
            
            return {
                'matched': matched,
                'confidence': confidence,
                'distance': best_distance,
                'ref_idx': best_ref_idx,
                'method': 'phash_regions',
                'best_region': best_region_name,
                'adaptive_threshold': adaptive_threshold,
                'all_distances': all_distances
            }
            
        except Exception as e:
            print(f"[WARNING] pHash matching failed: {e}")
            return {'matched': False, 'confidence': 0.0, 'ref_idx': -1, 'method': 'phash'}
    
    def detect_logo_in_screenshot(self, screenshot_path):
        """
        Detect logo using ensemble approach: combines template matching, DINOv2, and pHash.
        Uses weighted voting with adaptive thresholds.
        """
        if not Path(screenshot_path).exists():
            return {
                'logo_detected': 'No',
                'detection_method': 'none',
                'confidence': 0.0,
                'num_matches': 0,
                'best_ref_idx': -1
            }
        
        print(f"[DEBUG] Checking screenshot against {len(self.reference_logos)} reference logo(s)...")
        print(f"[DEBUG] Using ensemble approach: Template Matching + DINOv2 + pHash")
        
        # Run all three methods
        template_result = self.match_template_multi_scale(screenshot_path)
        dinov2_result = self.match_logo_dinov2(screenshot_path)
        phash_result = self.match_logo_phash(screenshot_path)
        
        template_matched = template_result.get('matched', False)
        template_conf = template_result.get('confidence', 0.0)
        template_ref_idx = template_result.get('ref_idx', -1)
        
        dinov2_matched = dinov2_result.get('matched', False)
        dinov2_conf = dinov2_result.get('confidence', 0.0)
        dinov2_ref_idx = dinov2_result.get('ref_idx', -1)
        
        phash_matched = phash_result.get('matched', False)
        phash_conf = phash_result.get('confidence', 0.0)
        phash_ref_idx = phash_result.get('ref_idx', -1)
        
        print(f"[DEBUG] Template Matching: matched={template_matched}, conf={template_conf:.4f}, ref={template_ref_idx}")
        print(f"[DEBUG] DINOv2: matched={dinov2_matched}, conf={dinov2_conf:.4f}, ref={dinov2_ref_idx}")
        print(f"[DEBUG] pHash: matched={phash_matched}, conf={phash_conf:.4f}, ref={phash_ref_idx}")
        
        # Ensemble voting with weighted confidence
        # Method weights: Template (high weight - very accurate), DINOv2 (medium), pHash (lower)
        method_weights = {
            'template': 1.5,  # Template matching is very reliable for exact matches
            'dinov2': 1.0,    # DINOv2 is good for semantic similarity
            'phash': 0.7      # pHash is good but can have false positives
        }
        
        # Count votes and calculate weighted confidence
        votes = []
        weighted_confidence = 0.0
        total_weight = 0.0
        
        if template_matched and template_ref_idx >= 0:
            votes.append(('template', template_ref_idx, template_conf))
            weighted_confidence += template_conf * method_weights['template']
            total_weight += method_weights['template']
        
        if dinov2_matched and dinov2_ref_idx >= 0:
            votes.append(('dinov2', dinov2_ref_idx, dinov2_conf))
            weighted_confidence += dinov2_conf * method_weights['dinov2']
            total_weight += method_weights['dinov2']
        
        if phash_matched and phash_ref_idx >= 0:
            votes.append(('phash', phash_ref_idx, phash_conf))
            weighted_confidence += phash_conf * method_weights['phash']
            total_weight += method_weights['phash']
        
        # Normalize weighted confidence
        if total_weight > 0:
            weighted_confidence = weighted_confidence / total_weight
        else:
            weighted_confidence = max(template_conf, dinov2_conf, phash_conf)
        
        # EXTREMELY STRICT Decision logic: Logo detected ONLY if:
        # ALL 3 methods MUST agree on the SAME reference logo with HIGH confidence
        # No exceptions, no single-method detections, no 2-method consensus
        
        logo_detected = False
        detection_method = 'none'
        best_ref_idx = -1
        
        # ONLY accept if ALL 3 methods agree on same reference with high confidence
        if len(votes) == 3:
            ref_indices = [v[1] for v in votes]
            confidences = [v[2] for v in votes]
            
            # Check if all agree on same reference
            if len(set(ref_indices)) == 1:
                # All agree on same reference - now check confidence requirements
                # Require ALL methods to have high confidence (not just above threshold)
                min_confidence_required = 0.85
                
                if all(conf >= min_confidence_required for conf in confidences):
                    # All 3 methods agree AND all have high confidence
                    logo_detected = True
                    best_ref_idx = ref_indices[0]
                    methods_used = '+'.join([v[0] for v in votes])
                    detection_method = f'ensemble_all3_{methods_used}'
                    print(f"[DEBUG] ✓✓ STRONG Consensus: ALL 3 methods agree on reference logo {best_ref_idx}")
                    print(f"[DEBUG]   Template: {template_conf:.4f}, DINOv2: {dinov2_conf:.4f}, pHash: {phash_conf:.4f}")
                else:
                    print(f"[DEBUG] ✗ All 3 agree but confidence too low:")
                    print(f"[DEBUG]   Template: {template_conf:.4f}, DINOv2: {dinov2_conf:.4f}, pHash: {phash_conf:.4f}")
                    print(f"[DEBUG]   Required: all >= {min_confidence_required}")
            else:
                print(f"[DEBUG] ✗ All 3 methods matched but on different references: {ref_indices}")
        else:
            print(f"[DEBUG] ✗ Not all 3 methods matched: template={template_matched}, dinov2={dinov2_matched}, phash={phash_matched}")
            print(f"[DEBUG]   Requiring ALL 3 methods to agree - no exceptions")
        
        # Set result
        result = {
            'logo_detected': 'Yes' if logo_detected else 'No',
            'detection_method': detection_method,
            'confidence': weighted_confidence if logo_detected else max(template_conf, dinov2_conf, phash_conf),
            'num_matches': int(weighted_confidence * 100) if logo_detected else 0,
            'best_ref_idx': best_ref_idx
        }
        
        # Calculate risk score
        risk_data = self.calculate_risk_score(result)
        result['risk_score'] = risk_data['score']
        result['risk_level'] = risk_data['level']
        
        if logo_detected:
            print(f"[DEBUG] ✓✓ Logo DETECTED: Reference {best_ref_idx}, method={detection_method}, conf={result['confidence']:.4f}, risk={result['risk_score']}/100")
        else:
            print(f"[DEBUG] ✗ Logo NOT detected: No consensus or high-confidence match")
        
        return result
    
    def calculate_risk_score(self, detection_result: Dict) -> Dict:
        """
        Calculate risk score based on logo detection results.
        
        Args:
            detection_result: Dictionary with detection results
            
        Returns:
            Dictionary with risk_score (0-100), risk_level, and factors
        """
        logo_detected = detection_result.get('logo_detected', 'No') == 'Yes'
        confidence = detection_result.get('confidence', 0.0)
        detection_method = detection_result.get('detection_method', 'none')
        
        risk_score = 0
        factors = []
        
        if logo_detected:
            # Base risk: Logo detected is a strong indicator
            risk_score = 50  # Base 50 points for logo detection
            
            # Confidence-based risk (0-30 points)
            # Higher confidence = higher risk (more certain it's the brand logo)
            confidence_points = min(confidence * 30, 30)
            risk_score += confidence_points
            factors.append(f"Logo detected with {confidence:.1%} confidence (+{confidence_points:.1f} points)")
            
            # Detection method risk (0-20 points)
            # All 3 methods = highest risk (strongest evidence)
            if 'all3' in detection_method or 'ensemble_all3' in detection_method:
                risk_score += 20
                factors.append("All 3 detection methods agreed (+20 points)")
            elif 'consensus' in detection_method:
                risk_score += 15
                factors.append("Multiple methods consensus (+15 points)")
            elif 'template' in detection_method:
                risk_score += 10
                factors.append("Template matching detected (+10 points)")
            elif 'dinov2' in detection_method:
                risk_score += 8
                factors.append("DINOv2 semantic similarity detected (+8 points)")
            elif 'phash' in detection_method:
                risk_score += 5
                factors.append("Perceptual hash match detected (+5 points)")
            
            # Very high confidence bonus (0-10 points)
            if confidence >= 0.95:
                risk_score += 10
                factors.append("Extremely high confidence match (+10 points)")
            elif confidence >= 0.90:
                risk_score += 5
                factors.append("Very high confidence match (+5 points)")
        else:
            # No logo detected = low risk
            risk_score = 0
            factors.append("No brand logo detected (0 points)")
        
        # Normalize to 0-100
        risk_score = min(risk_score, 100)
        
        # Determine risk level
        if risk_score >= 80:
            risk_level = 'Critical'
        elif risk_score >= 60:
            risk_level = 'High'
        elif risk_score >= 40:
            risk_level = 'Medium'
        elif risk_score >= 20:
            risk_level = 'Low'
        else:
            risk_level = 'Very Low'
        
        return {
            'score': round(risk_score, 2),
            'level': risk_level,
            'factors': factors
        }
    
    def process_dataset(self, df, screenshot_base_path=None):
        """Process the entire dataset and add logo detection columns."""
        df['logo_detected'] = pd.Series(dtype='object')
        df['detection_method'] = pd.Series(dtype='object')
        df['confidence'] = pd.Series(dtype='float64')
        df['num_matches'] = pd.Series(dtype='int64')
        df['risk_score'] = pd.Series(dtype='float64')
        df['risk_level'] = pd.Series(dtype='object')
        
        if screenshot_base_path is None:
            script_dir = Path(__file__).parent.absolute()
            project_root = script_dir.parent
            screenshot_base_path = project_root
        
        total_rows = len(df)
        processed = 0
        skipped_real = 0
        
        print(f"\n[INFO] Processing {total_rows} domains for logo detection...")
        print(f"[INFO] Using ensemble approach: Template Matching + DINOv2 + pHash")
        print(f"[INFO]   - Multi-scale template matching (OpenCV)")
        print(f"[INFO]   - DINOv2 semantic similarity (adaptive thresholds)")
        print(f"[INFO]   - pHash validation (adaptive thresholds)")
        print(f"[INFO]   - Ensemble voting with weighted confidence")
        print()
        
        for idx, row in df.iterrows():
            domain = row['domain']
            real_brand = str(row.get('real_brand', 'no')).lower().strip()
            
            if real_brand == 'yes':
                skipped_real += 1
                df.at[idx, 'logo_detected'] = 'No'
                continue
            
            screenshot_path = row.get('screenshot', '')
            if not screenshot_path:
                print(f"[WARNING] No screenshot path for {domain}, skipping...")
                continue
            
            screenshot_path_normalized = str(screenshot_path).replace('\\', '/')
            if not Path(screenshot_path_normalized).is_absolute():
                path_parts = screenshot_path_normalized.split('/')
                full_screenshot_path = screenshot_base_path
                for part in path_parts:
                    if part:
                        full_screenshot_path = full_screenshot_path / part
            else:
                full_screenshot_path = Path(screenshot_path_normalized)
            
            processed += 1
            print(f"[{processed}/{total_rows - skipped_real}] Processing {domain}...")
            
            try:
                if not full_screenshot_path.exists():
                    print(f"[WARNING] Screenshot not found: {full_screenshot_path}, skipping...")
                    df.at[idx, 'logo_detected'] = 'No'
                    df.at[idx, 'detection_method'] = 'none'
                    df.at[idx, 'confidence'] = 0.0
                    df.at[idx, 'num_matches'] = 0
                    df.at[idx, 'risk_score'] = 0.0
                    df.at[idx, 'risk_level'] = 'Very Low'
                    continue
                
                detection_result = self.detect_logo_in_screenshot(full_screenshot_path)
                
                df.at[idx, 'logo_detected'] = detection_result.get('logo_detected', 'No')
                df.at[idx, 'detection_method'] = detection_result.get('detection_method', 'none')
                df.at[idx, 'confidence'] = detection_result.get('confidence', 0.0)
                df.at[idx, 'num_matches'] = detection_result.get('num_matches', 0)
                df.at[idx, 'risk_score'] = detection_result.get('risk_score', 0.0)
                df.at[idx, 'risk_level'] = detection_result.get('risk_level', 'Very Low')
                
                detected = detection_result.get('logo_detected', 'No')
                conf = detection_result.get('confidence', 0.0)
                method = detection_result.get('detection_method', 'none')
                risk_score = detection_result.get('risk_score', 0.0)
                risk_level = detection_result.get('risk_level', 'Very Low')
                print(f"  Result: {detected} (method: {method}, confidence: {conf:.4f}, risk: {risk_score}/100 ({risk_level}))")
            except Exception as e:
                print(f"[ERROR] Failed to process {domain}: {e}")
                import traceback
                traceback.print_exc()
                df.at[idx, 'logo_detected'] = 'No'
                df.at[idx, 'detection_method'] = 'error'
                df.at[idx, 'confidence'] = 0.0
                df.at[idx, 'num_matches'] = 0
                df.at[idx, 'risk_score'] = 0.0
                df.at[idx, 'risk_level'] = 'Very Low'
        
        print(f"\n[INFO] Processing complete!")
        print(f"[INFO] Processed: {processed} domains")
        print(f"[INFO] Skipped (real brand): {skipped_real} domains")
        
        import numpy as np
        df = df.replace([np.nan, pd.NA, pd.NaT, float('nan')], None)
        df = df.replace(['nan', 'NaN', 'None'], None)
        
        for col in df.columns:
            mask = df[col].isna()
            if mask.any():
                df.loc[mask, col] = None
        
        return df
    
    @staticmethod
    def generate_report(results_data, brand_domain, output_path=None):
        """
        Generate HTML report for logo detection results.
        
        Args:
            results_data: List of detection results (from DataFrame.to_dict('records'))
            brand_domain: Brand domain name
            output_path: Optional path to save report file
            
        Returns:
            Report as string (HTML format)
        """
        from datetime import datetime
        from typing import Optional
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Calculate summary statistics
        total_domains = len(results_data)
        logos_detected = sum(1 for r in results_data if r.get('logo_detected') == 'Yes')
        no_logos = total_domains - logos_detected
        
        # Risk score distribution
        critical = sum(1 for r in results_data if r.get('risk_score', 0) >= 80)
        high = sum(1 for r in results_data if 60 <= r.get('risk_score', 0) < 80)
        medium = sum(1 for r in results_data if 40 <= r.get('risk_score', 0) < 60)
        low = sum(1 for r in results_data if 20 <= r.get('risk_score', 0) < 40)
        very_low = sum(1 for r in results_data if r.get('risk_score', 0) < 20)
        
        # Average risk score
        avg_risk = sum(r.get('risk_score', 0) for r in results_data) / total_domains if total_domains > 0 else 0
        
        # Determine overall risk color
        if avg_risk >= 70:
            risk_color = '#dc3545'  # Red
        elif avg_risk >= 50:
            risk_color = '#fd7e14'  # Orange
        elif avg_risk >= 30:
            risk_color = '#ffc107'  # Yellow
        elif avg_risk >= 10:
            risk_color = '#0dcaf0'  # Cyan
        else:
            risk_color = '#198754'  # Green
        
        # Build threats table
        threats_html = ""
        threats_found = [r for r in results_data if r.get('logo_detected') == 'Yes']
        
        if threats_found:
            for threat in threats_found[:50]:  # Limit to 50 for display
                domain = threat.get('domain', 'Unknown')
                confidence = threat.get('confidence', 0.0)
                method = threat.get('detection_method', 'unknown')
                risk_score = threat.get('risk_score', 0.0)
                risk_level = threat.get('risk_level', 'Unknown')
                
                threats_html += f"""
                <tr>
                    <td>{domain}</td>
                    <td>{confidence:.2%}</td>
                    <td>{method}</td>
                    <td style="color: {risk_color}"><strong>{risk_score:.1f}/100</strong></td>
                    <td>{risk_level}</td>
                </tr>
                """
        else:
            threats_html = '<tr><td colspan="5" style="text-align: center; color: #198754;">No logos detected - All domains are clean!</td></tr>'
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Logo Detection Report - {brand_domain}</title>
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
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .summary-card {{
            background: #f9f9f9;
            padding: 20px;
            border-radius: 5px;
            text-align: center;
            border-left: 4px solid {risk_color};
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
        }}
        .summary-card .value {{
            font-size: 32px;
            font-weight: bold;
            color: {risk_color};
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: {risk_color};
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
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
        <h1>Logo Detection Report</h1>
        <div style="text-align: center; margin: 20px 0;">
            <h2 style="color: #666;">Brand: <strong>{brand_domain}</strong></h2>
            <p style="color: #999;">Generated: {timestamp}</p>
        </div>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Total Domains Analyzed</h3>
                <div class="value">{total_domains}</div>
            </div>
            <div class="summary-card">
                <h3>Logos Detected</h3>
                <div class="value" style="color: #dc3545;">{logos_detected}</div>
            </div>
            <div class="summary-card">
                <h3>Clean Domains</h3>
                <div class="value" style="color: #198754;">{no_logos}</div>
            </div>
            <div class="summary-card">
                <h3>Average Risk Score</h3>
                <div class="value">{avg_risk:.1f}/100</div>
            </div>
        </div>
        
        <div style="margin: 30px 0;">
            <h2>Risk Score Distribution</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>Critical (80-100)</h3>
                    <div class="value" style="color: #dc3545;">{critical}</div>
                </div>
                <div class="summary-card">
                    <h3>High (60-79)</h3>
                    <div class="value" style="color: #fd7e14;">{high}</div>
                </div>
                <div class="summary-card">
                    <h3>Medium (40-59)</h3>
                    <div class="value" style="color: #ffc107;">{medium}</div>
                </div>
                <div class="summary-card">
                    <h3>Low (20-39)</h3>
                    <div class="value" style="color: #0dcaf0;">{low}</div>
                </div>
                <div class="summary-card">
                    <h3>Very Low (0-19)</h3>
                    <div class="value" style="color: #198754;">{very_low}</div>
                </div>
            </div>
        </div>
        
        <div style="margin: 30px 0;">
            <h2>Detected Threats</h2>
            <table>
                <thead>
                    <tr>
                        <th>Domain</th>
                        <th>Confidence</th>
                        <th>Detection Method</th>
                        <th>Risk Score</th>
                        <th>Risk Level</th>
                    </tr>
                </thead>
                <tbody>
                    {threats_html}
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>Report generated by Brand Protection System</p>
            <p>Detection Method: Multi-Method Ensemble (Template Matching + DINOv2 + pHash)</p>
        </div>
    </div>
</body>
</html>
"""
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
        
        return html


def find_reference_logos(brand_domain, project_root):
    """Find reference logo files for a brand."""
    brand_name = get_brand_name_from_domain(brand_domain)
    logo_dir = project_root / "data" / "reference_logos" / brand_name
    
    if not logo_dir.exists():
        raise FileNotFoundError(
            f"Reference logo folder not found: {logo_dir}\n"
            f"Please create the folder '{brand_name}' inside 'data/reference_logos/' and add logo images."
        )
    
    if not logo_dir.is_dir():
        raise FileNotFoundError(f"Path exists but is not a directory: {logo_dir}")
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.webp'}
    logo_files = []
    
    for file_path in logo_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            logo_files.append(file_path)
    
    logo_files.sort(key=lambda x: x.name.lower())
    
    if not logo_files:
        raise FileNotFoundError(
            f"No image files found in {logo_dir}\n"
            f"Please add logo images (PNG, JPG, JPEG, etc.) to the folder."
        )
    
    print(f"[INFO] Found {len(logo_files)} reference logo image(s) in {logo_dir}:")
    for logo_file in logo_files:
        print(f"[INFO]   - {logo_file.name}")
    
    return logo_files


def get_brand_name_from_domain(domain):
    """Extract brand name from domain."""
    try:
        import tldextract
        ext = tldextract.extract(domain)
        return ext.domain.lower()
    except:
        domain_clean = domain.lower().replace('www.', '').replace('http://', '').replace('https://', '')
        if '.' in domain_clean:
            return domain_clean.split('.')[0]
        return domain_clean


def main():
    parser = argparse.ArgumentParser(description='Advanced logo detection using ensemble approach')
    parser.add_argument('--input', type=str, default='dataset_final.csv',
                       help='Input CSV file with domain and screenshot information')
    parser.add_argument('--output', type=str, default='dataset_with_logo.csv',
                       help='Output CSV file with logo detection results')
    parser.add_argument('--brand-domain', type=str,
                       help='Brand domain (e.g., facebook.com)')
    parser.add_argument('--reference-logos', type=str, nargs='+',
                       help='Paths to reference logo images')
    parser.add_argument('--similarity-threshold', type=float, default=0.75,
                       help='Base DINOv2 similarity threshold (adaptive)')
    parser.add_argument('--phash-threshold', type=int, default=5,
                       help='Base pHash distance threshold (adaptive)')
    parser.add_argument('--template-threshold', type=float, default=0.7,
                       help='Template matching threshold (0.0-1.0)')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    
    try:
        if args.reference_logos:
            reference_logo_paths = [Path(p) for p in args.reference_logos]
            print(f"[INFO] Using {len(reference_logo_paths)} explicitly provided reference logo(s)")
        elif args.brand_domain:
            print(f"[INFO] Looking for reference logos for brand: {args.brand_domain}")
            reference_logo_paths = find_reference_logos(args.brand_domain, project_root)
        else:
            input_path = project_root / "src" / args.input
            if not input_path.exists():
                input_path = Path(args.input)
            
            if input_path.exists():
                df_temp = pd.read_csv(input_path, encoding='utf-8', nrows=10)
                if 'brand_domain' in df_temp.columns and not df_temp['brand_domain'].empty:
                    brand_domain_from_csv = df_temp['brand_domain'].iloc[0]
                    print(f"[INFO] Inferring brand domain from CSV: {brand_domain_from_csv}")
                    reference_logo_paths = find_reference_logos(brand_domain_from_csv, project_root)
                else:
                    print("[ERROR] Cannot infer brand domain from CSV.")
                    sys.exit(1)
            else:
                print(f"[ERROR] Input file not found: {input_path}")
                sys.exit(1)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    try:
        detector = LogoDetector(
            reference_logo_paths,
            similarity_threshold=args.similarity_threshold,
            phash_threshold=args.phash_threshold,
            template_threshold=args.template_threshold
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize logo detector: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    try:
        input_path = project_root / "src" / args.input
        if not input_path.exists():
            input_path = Path(args.input)
        
        df = pd.read_csv(input_path, encoding='utf-8')
        df_result = detector.process_dataset(df, screenshot_base_path=project_root)
        
        output_path = project_root / "src" / args.output
        df_result.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n[INFO] Results saved to: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to process dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
