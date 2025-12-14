#!/usr/bin/env python3
import pandas as pd
import argparse
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("http_file", help="CSV from http_info.py")
    parser.add_argument("--output_file", default="dataset_final.csv")
    args = parser.parse_args()

    # Get proper paths
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    
    http_file_path = Path(args.http_file)
    if not http_file_path.is_absolute():
        http_file_path = script_dir / args.http_file
    
    if not http_file_path.exists():
        print(f"[ERROR] HTTP file not found: {http_file_path}")
        return
    
    df = pd.read_csv(http_file_path)

    # Fixed screenshot folder path - use proper path handling
    screenshot_folder = project_root / "data" / "screenshots"

    # Add screenshot paths (preserve all existing columns)
    screenshot_paths = []
    for domain in df['domain']:
        file_path = screenshot_folder / f"{domain.replace('.','_')}.png"
        if file_path.exists():
            # Use relative path from project root for portability
            screenshot_paths.append(str(file_path.relative_to(project_root)))
        else:
            screenshot_paths.append('')
    
    # Only add screenshot column if it doesn't exist
    if 'screenshot' not in df.columns:
        df['screenshot'] = screenshot_paths
    else:
        # Update existing screenshot paths
        for idx, path in enumerate(screenshot_paths):
            if path and (not df.loc[idx, 'screenshot'] or df.loc[idx, 'screenshot'] == ''):
                df.loc[idx, 'screenshot'] = path

    # Ensure basic columns are present (legitimacy columns are optional)
    required_columns = [
        'domain', 'ip', 'http_status', 'final_url', 'title',
        'ssl_issuer', 'ssl_notBefore', 'ssl_notAfter', 'screenshot'
    ]
    
    # Add missing columns with empty values
    for col in required_columns:
        if col not in df.columns:
            df[col] = ''

    output_path = script_dir / args.output_file
    df.to_csv(output_path, index=False)
    
    # Print summary
    print(f"\n[DONE] Final dataset saved to {output_path}")
    print(f"Total domains: {len(df)}")
    print(f"Domains with screenshots: {df['screenshot'].astype(bool).sum()}")
    if 'is_legitimate' in df.columns:
        print(f"Legitimate domains: {df['is_legitimate'].astype(bool).sum()}")
    print(f"Columns: {', '.join(df.columns.tolist())}")

if __name__ == "__main__":
    main()

