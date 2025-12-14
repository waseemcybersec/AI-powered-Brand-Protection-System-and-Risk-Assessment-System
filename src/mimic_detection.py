#!/usr/bin/env python3
"""
Brand Mimic Detection Script

This script analyzes candidate domain screenshots to detect if they mimic
the official brand by comparing visual similarity using:
1. Perceptual Hash (pHash) distance
2. CLIP image embeddings cosine similarity

Usage:
    python mimic_detection.py [--input dataset_final.csv] [--output dataset_with_mimic.csv]
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import imagehash

# CLIP imports
try:
    import torch
    import torchvision.transforms as transforms
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("[WARNING] CLIP libraries not available. Install with: pip install torch transformers")
    print("[INFO] Will only compute pHash similarity.")


class MimicDetector:
    """Detects brand mimicry using visual similarity metrics."""
    
    def __init__(self, reference_image_paths, phash_threshold=10, clip_threshold=0.80):
        """
        Initialize the mimic detector.
        
        Args:
            reference_image_paths: Path or list of paths to reference brand screenshots
            phash_threshold: Maximum pHash distance to consider a mimic (default: 10)
            clip_threshold: Minimum CLIP similarity to consider a mimic (default: 0.80)
        """
        # Handle single path or list of paths
        if isinstance(reference_image_paths, (str, Path)):
            reference_image_paths = [reference_image_paths]
        
        self.reference_image_paths = [Path(p) for p in reference_image_paths]
        self.phash_threshold = phash_threshold
        self.clip_threshold = clip_threshold
        
        # Load all reference images
        self.reference_images = []
        self.reference_phashes = []
        
        for ref_path in self.reference_image_paths:
            if not ref_path.exists():
                print(f"[WARNING] Reference image not found: {ref_path}, skipping...")
                continue
            
            print(f"[INFO] Loading reference image: {ref_path}")
            ref_image = Image.open(ref_path).convert('RGB')
            self.reference_images.append(ref_image)
            
            # Compute reference pHash
            ref_phash = imagehash.phash(ref_image)
            self.reference_phashes.append(ref_phash)
            print(f"[INFO] Reference pHash computed: {ref_phash}")
        
        if not self.reference_images:
            raise FileNotFoundError("No valid reference images found")
        
        print(f"[INFO] Loaded {len(self.reference_images)} reference image(s)")
        
        # Initialize CLIP model if available
        self.clip_model = None
        self.clip_processor = None
        self.reference_clip_embeddings = []
        self.clip_available = CLIP_AVAILABLE  # Store as instance variable
        
        if self.clip_available:
            try:
                print("[INFO] Loading CLIP model...")
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                
                # Compute reference CLIP embeddings for all reference images
                print("[INFO] Computing reference CLIP embeddings...")
                for ref_image in self.reference_images:
                    inputs = self.clip_processor(images=ref_image, return_tensors="pt")
                    with torch.no_grad():
                        ref_embedding = self.clip_model.get_image_features(**inputs)
                        # Normalize the embedding
                        ref_embedding = ref_embedding / ref_embedding.norm(dim=-1, keepdim=True)
                        self.reference_clip_embeddings.append(ref_embedding)
                print(f"[INFO] CLIP model ready. Computed {len(self.reference_clip_embeddings)} reference embeddings.")
            except Exception as e:
                print(f"[WARNING] Failed to load CLIP model: {e}")
                print("[INFO] Will only compute pHash similarity.")
                self.clip_available = False
    
    def compute_phash_distance(self, image_path):
        """
        Compute perceptual hash distance between candidate and best matching reference.
        Compares candidate against ALL reference images one by one and returns the best match.
        
        Args:
            image_path: Path to candidate screenshot
            
        Returns:
            Best (minimum) pHash distance (0 = identical, higher = more different)
        """
        try:
            if not Path(image_path).exists():
                return None
            
            candidate_image = Image.open(image_path).convert('RGB')
            candidate_phash = imagehash.phash(candidate_image)
            
            # Compare against ALL references one by one
            # For each reference: ref1, ref2, ref3, etc.
            distances = []
            for i, ref_phash in enumerate(self.reference_phashes):
                distance = int(ref_phash - candidate_phash)
                distances.append(distance)
            
            # Take the best (minimum) distance - if ANY reference matches, this will be low
            best_distance = min(distances)
            
            return best_distance
        except Exception as e:
            print(f"[WARNING] Failed to compute pHash for {image_path}: {e}")
            return None
    
    def compute_clip_similarity(self, image_path):
        """
        Compute CLIP cosine similarity between candidate and best matching reference.
        Compares candidate against ALL reference images and returns the best match.
        
        Args:
            image_path: Path to candidate screenshot
            
        Returns:
            Best (maximum) CLIP cosine similarity (0-1, higher = more similar)
        """
        if not self.clip_available or self.clip_model is None:
            return None
        
        try:
            if not Path(image_path).exists():
                return None
            
            candidate_image = Image.open(image_path).convert('RGB')
            inputs = self.clip_processor(images=candidate_image, return_tensors="pt")
            
            with torch.no_grad():
                candidate_embedding = self.clip_model.get_image_features(**inputs)
                # Normalize the embedding
                candidate_embedding = candidate_embedding / candidate_embedding.norm(dim=-1, keepdim=True)
                
                # Compare against ALL references one by one
                # For each reference: ref1, ref2, ref3, etc.
                similarities = []
                for i, ref_embedding in enumerate(self.reference_clip_embeddings):
                    similarity = torch.nn.functional.cosine_similarity(
                        ref_embedding, 
                        candidate_embedding
                    ).item()
                    similarities.append(similarity)
                
                # Take the best (maximum) similarity - if ANY reference matches, this will be high
                best_similarity = max(similarities)
            
            return float(best_similarity)
        except Exception as e:
            print(f"[WARNING] Failed to compute CLIP similarity for {image_path}: {e}")
            return None
    
    def detect_mimic(self, phash_distance, clip_similarity):
        """
        Determine if a domain is mimicking the brand.
        BOTH metrics must pass: pHash distance <= threshold AND CLIP similarity >= threshold.
        
        Args:
            phash_distance: pHash distance (must be <= threshold)
            clip_similarity: CLIP similarity (must be >= threshold)
            
        Returns:
            "Yes" if mimicking (BOTH pass), "No" otherwise
        """
        # Both metrics must be available and pass
        if phash_distance is not None and clip_similarity is not None:
            phash_passes = phash_distance <= self.phash_threshold
            clip_passes = clip_similarity >= self.clip_threshold
            
            # BOTH must pass to be considered a mimic
            if phash_passes and clip_passes:
                return "Yes"
            return "No"
        
        # If only CLIP is available (pHash failed)
        if clip_similarity is not None:
            # Still require CLIP to pass, but note that pHash wasn't available
            if clip_similarity >= self.clip_threshold:
                return "Yes"
            return "No"
        
        # If only pHash is available (CLIP failed)
        if phash_distance is not None:
            if phash_distance <= self.phash_threshold:
                return "Yes"
            return "No"
        
        # If neither is available
        return "No"
    
    def calculate_risk_score(self, phash_distance, clip_similarity, mimic_result):
        """
        Calculate risk score based on mimic detection results.
        
        Args:
            phash_distance: pHash distance (lower = more similar)
            clip_similarity: CLIP similarity (higher = more similar)
            mimic_result: "Yes" if mimicking, "No" otherwise
            
        Returns:
            Dictionary with risk_score (0-100), risk_level, and factors
        """
        from typing import Dict
        
        is_mimic = mimic_result == "Yes"
        risk_score = 0
        factors = []
        
        if is_mimic:
            # Base risk: Mimic detected is a strong indicator
            risk_score = 40  # Base 40 points for mimic detection
            
            # pHash-based risk (0-25 points)
            # Lower distance = higher similarity = higher risk
            if phash_distance is not None:
                # Distance 0-5: very similar (high risk)
                # Distance 6-10: similar (medium risk)
                if phash_distance <= 5:
                    phash_points = 25
                    factors.append(f"Very high visual similarity (pHash: {phash_distance}, +25 points)")
                elif phash_distance <= 8:
                    phash_points = 20
                    factors.append(f"High visual similarity (pHash: {phash_distance}, +20 points)")
                elif phash_distance <= 10:
                    phash_points = 15
                    factors.append(f"Moderate visual similarity (pHash: {phash_distance}, +15 points)")
                else:
                    phash_points = 10
                    factors.append(f"Some visual similarity (pHash: {phash_distance}, +10 points)")
                risk_score += phash_points
            
            # CLIP similarity-based risk (0-25 points)
            # Higher similarity = higher risk
            if clip_similarity is not None:
                if clip_similarity >= 0.90:
                    clip_points = 25
                    factors.append(f"Extremely high semantic similarity (CLIP: {clip_similarity:.3f}, +25 points)")
                elif clip_similarity >= 0.85:
                    clip_points = 20
                    factors.append(f"Very high semantic similarity (CLIP: {clip_similarity:.3f}, +20 points)")
                elif clip_similarity >= 0.80:
                    clip_points = 15
                    factors.append(f"High semantic similarity (CLIP: {clip_similarity:.3f}, +15 points)")
                else:
                    clip_points = 10
                    factors.append(f"Moderate semantic similarity (CLIP: {clip_similarity:.3f}, +10 points)")
                risk_score += clip_points
            
            # Both methods agree bonus (0-10 points)
            if phash_distance is not None and clip_similarity is not None:
                phash_passes = phash_distance <= self.phash_threshold
                clip_passes = clip_similarity >= self.clip_threshold
                if phash_passes and clip_passes:
                    risk_score += 10
                    factors.append("Both pHash and CLIP methods confirmed mimic (+10 points)")
        else:
            # No mimic detected = low risk
            risk_score = 0
            factors.append("No visual mimicry detected (0 points)")
        
        # Normalize to 0-100
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
            'factors': factors
        }
    
    def process_dataset(self, df, screenshot_base_path=None):
        """
        Process the entire dataset and add mimic detection columns.
        
        Args:
            df: DataFrame with domain and screenshot information
            screenshot_base_path: Base path for screenshots (if relative paths in CSV)
            
        Returns:
            DataFrame with added columns: phash_distance, clip_similarity, mimic_brand
        """
        # Add new columns with proper dtype to avoid NaN issues
        df['phash_distance'] = pd.Series(dtype='object')
        df['clip_similarity'] = pd.Series(dtype='object')
        df['mimic_brand'] = pd.Series(dtype='object')
        df['risk_score'] = pd.Series(dtype='float64')
        df['risk_level'] = pd.Series(dtype='object')
        
        # Get base path for screenshots
        if screenshot_base_path is None:
            script_dir = Path(__file__).parent.absolute()
            project_root = script_dir.parent
            screenshot_base_path = project_root
        
        total_rows = len(df)
        processed = 0
        skipped_real = 0
        
        print(f"\n[INFO] Processing {total_rows} domains...")
        print(f"[INFO] pHash threshold: {self.phash_threshold}")
        if self.clip_available:
            print(f"[INFO] CLIP similarity threshold: {self.clip_threshold}")
        print()
        
        for idx, row in df.iterrows():
            domain = row['domain']
            real_brand = str(row.get('real_brand', 'no')).lower().strip()
            
            # Skip real brand domains
            if real_brand == 'yes':
                skipped_real += 1
                df.at[idx, 'mimic_brand'] = 'No'  # Real brands are not mimics
                df.at[idx, 'risk_score'] = 0.0
                df.at[idx, 'risk_level'] = 'Very Low'
                continue
            
            # Get screenshot path
            screenshot_path = row.get('screenshot', '')
            if not screenshot_path:
                print(f"[WARNING] No screenshot path for {domain}, skipping...")
                continue
            
            # Handle relative paths
            # Normalize path separators (handle both / and \)
            screenshot_path_normalized = str(screenshot_path).replace('\\', '/')
            if not Path(screenshot_path_normalized).is_absolute():
                # Split and rejoin to handle pathlib properly
                path_parts = screenshot_path_normalized.split('/')
                full_screenshot_path = screenshot_base_path
                for part in path_parts:
                    if part:  # Skip empty parts
                        full_screenshot_path = full_screenshot_path / part
            else:
                full_screenshot_path = Path(screenshot_path_normalized)
            
            processed += 1
            print(f"[{processed}/{total_rows - skipped_real}] Processing {domain}...")
            print(f"  Comparing against {len(self.reference_images)} reference image(s)...")
            
            # Compute pHash distance (compares against ALL references)
            phash_dist = self.compute_phash_distance(full_screenshot_path)
            if phash_dist is not None:
                df.at[idx, 'phash_distance'] = phash_dist
            else:
                df.at[idx, 'phash_distance'] = None
            
            # Compute CLIP similarity (compares against ALL references)
            clip_sim = self.compute_clip_similarity(full_screenshot_path)
            if clip_sim is not None:
                df.at[idx, 'clip_similarity'] = clip_sim
            else:
                df.at[idx, 'clip_similarity'] = None
            
            # Detect mimic - if ANY reference matches, mark as mimic
            mimic_result = self.detect_mimic(phash_dist, clip_sim)
            df.at[idx, 'mimic_brand'] = mimic_result if mimic_result else 'No'
            
            # Calculate risk score
            risk_data = self.calculate_risk_score(phash_dist, clip_sim, mimic_result)
            df.at[idx, 'risk_score'] = risk_data['score']
            df.at[idx, 'risk_level'] = risk_data['level']
            
            # Print results with both checks
            phash_str = str(phash_dist) if phash_dist is not None else 'N/A'
            clip_str = f"{clip_sim:.3f}" if clip_sim is not None else 'N/A'
            phash_pass = "✓" if (phash_dist is not None and phash_dist <= self.phash_threshold) else "✗"
            clip_pass = "✓" if (clip_sim is not None and clip_sim >= self.clip_threshold) else "✗"
            risk_score = risk_data['score']
            risk_level = risk_data['level']
            result_str = f"  Best match - pHash: {phash_str} (threshold: {self.phash_threshold}) {phash_pass}, CLIP: {clip_str} (threshold: {self.clip_threshold}) {clip_pass}, Mimic: {mimic_result}, Risk: {risk_score}/100 ({risk_level})"
            print(result_str)
        
        print(f"\n[INFO] Processing complete!")
        print(f"[INFO] Processed: {processed} domains")
        print(f"[INFO] Skipped (real brand): {skipped_real} domains")
        
        # Replace any remaining NaN/NaT values with None before returning
        # Handle different NaN types - use replace instead of fillna to avoid version issues
        import numpy as np
        # Replace all NaN types with None using replace (more reliable than fillna)
        df = df.replace([np.nan, pd.NA, pd.NaT, float('nan')], None)
        # Also replace string representations
        df = df.replace(['nan', 'NaN', 'None'], None)
        # For any remaining NaN values, set them to None explicitly
        for col in df.columns:
            mask = df[col].isna()
            if mask.any():
                df.loc[mask, col] = None
        
        return df
    
    @staticmethod
    def generate_report(results_data, brand_domain, output_path=None):
        """
        Generate HTML report for mimic detection results.
        
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
        mimics_detected = sum(1 for r in results_data if r.get('mimic_brand') == 'Yes')
        no_mimics = total_domains - mimics_detected
        
        # Risk score distribution
        critical = sum(1 for r in results_data if r.get('risk_score', 0) >= 70)
        high = sum(1 for r in results_data if 50 <= r.get('risk_score', 0) < 70)
        medium = sum(1 for r in results_data if 30 <= r.get('risk_score', 0) < 50)
        low = sum(1 for r in results_data if 10 <= r.get('risk_score', 0) < 30)
        very_low = sum(1 for r in results_data if r.get('risk_score', 0) < 10)
        
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
        threats_found = [r for r in results_data if r.get('mimic_brand') == 'Yes']
        
        if threats_found:
            for threat in threats_found[:50]:  # Limit to 50 for display
                domain = threat.get('domain', 'Unknown')
                phash_dist = threat.get('phash_distance')
                clip_sim = threat.get('clip_similarity')
                risk_score = threat.get('risk_score', 0.0)
                risk_level = threat.get('risk_level', 'Unknown')
                
                phash_str = f"{phash_dist}" if phash_dist is not None else "N/A"
                clip_str = f"{clip_sim:.3f}" if clip_sim is not None else "N/A"
                
                threats_html += f"""
                <tr>
                    <td>{domain}</td>
                    <td>{phash_str}</td>
                    <td>{clip_str}</td>
                    <td style="color: {risk_color}"><strong>{risk_score:.1f}/100</strong></td>
                    <td>{risk_level}</td>
                </tr>
                """
        else:
            threats_html = '<tr><td colspan="5" style="text-align: center; color: #198754;">No mimics detected - All domains are clean!</td></tr>'
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mimic Detection Report - {brand_domain}</title>
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
        <h1>Mimic Detection Report</h1>
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
                <h3>Mimics Detected</h3>
                <div class="value" style="color: #dc3545;">{mimics_detected}</div>
            </div>
            <div class="summary-card">
                <h3>Clean Domains</h3>
                <div class="value" style="color: #198754;">{no_mimics}</div>
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
                    <h3>Critical (70-100)</h3>
                    <div class="value" style="color: #dc3545;">{critical}</div>
                </div>
                <div class="summary-card">
                    <h3>High (50-69)</h3>
                    <div class="value" style="color: #fd7e14;">{high}</div>
                </div>
                <div class="summary-card">
                    <h3>Medium (30-49)</h3>
                    <div class="value" style="color: #ffc107;">{medium}</div>
                </div>
                <div class="summary-card">
                    <h3>Low (10-29)</h3>
                    <div class="value" style="color: #0dcaf0;">{low}</div>
                </div>
                <div class="summary-card">
                    <h3>Very Low (0-9)</h3>
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
                        <th>pHash Distance</th>
                        <th>CLIP Similarity</th>
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
            <p>Detection Method: Visual Similarity Analysis (CLIP + Perceptual Hashing)</p>
        </div>
    </div>
</body>
</html>
"""
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
        
        return html


def main():
    parser = argparse.ArgumentParser(
        description="Detect brand mimicry using visual similarity metrics"
    )
    parser.add_argument(
        '--input',
        default='dataset_final.csv',
        help='Input CSV file with domain and screenshot information (default: dataset_final.csv)'
    )
    parser.add_argument(
        '--output',
        default='dataset_with_mimic.csv',
        help='Output CSV file with mimic detection results (default: dataset_with_mimic.csv)'
    )
    parser.add_argument(
        '--reference',
        default=None,
        help='Path to reference screenshot (default: ../data/reference/homepage.png)'
    )
    parser.add_argument(
        '--phash-threshold',
        type=int,
        default=10,
        help='Maximum pHash distance to consider a mimic (default: 10, only used if CLIP unavailable)'
    )
    parser.add_argument(
        '--clip-threshold',
        type=float,
        default=0.80,
        help='Minimum CLIP similarity to consider a mimic (default: 0.80 = 80%)'
    )
    
    args = parser.parse_args()
    
    # Get script directory and project root
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    
    # Set default reference image path
    if args.reference is None:
        reference_path = project_root / "data" / "reference" / "homepage.png"
    else:
        reference_path = Path(args.reference)
    
    # Load input CSV
    input_path = script_dir / args.input
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return
    
    print(f"[INFO] Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"[INFO] Loaded {len(df)} rows")
    
    # Initialize mimic detector
    try:
        detector = MimicDetector(
            reference_path,
            phash_threshold=args.phash_threshold,
            clip_threshold=args.clip_threshold
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize mimic detector: {e}")
        return
    
    # Process dataset
    try:
        df_result = detector.process_dataset(df, screenshot_base_path=project_root)
    except Exception as e:
        print(f"[ERROR] Failed to process dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save results
    output_path = script_dir / args.output
    print(f"\n[INFO] Saving results to: {output_path}")
    df_result.to_csv(output_path, index=False)
    print(f"[✓] Results saved successfully!")
    
    # Print summary statistics
    print("\n[SUMMARY]")
    print(f"Total domains: {len(df_result)}")
    print(f"Domains marked as mimic: {len(df_result[df_result['mimic_brand'] == 'Yes'])}")
    print(f"Domains not mimicking: {len(df_result[df_result['mimic_brand'] == 'No'])}")
    if len(df_result[df_result['mimic_brand'] == 'Unknown']) > 0:
        print(f"Domains with unknown status: {len(df_result[df_result['mimic_brand'] == 'Unknown'])}")


if __name__ == "__main__":
    main()

