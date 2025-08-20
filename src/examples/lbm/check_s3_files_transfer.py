#!/usr/bin/env python3
"""
Script to check if S3 files exist in parallel.
Takes a file path containing S3 URIs (one per line) and checks existence without downloading.

Dependencies:
- boto3: AWS SDK for Python
- tqdm: Progress bar library
"""

import argparse
import sys
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from tqdm import tqdm


def parse_s3_uri(s3_uri):
    """Parse S3 URI to extract bucket and key."""
    parsed = urlparse(s3_uri)
    if parsed.scheme != 's3':
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    return bucket, key


def check_s3_file_exists(bucket, key, s3_client):
    """Check if a single S3 file exists."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            # Re-raise other client errors (access denied, etc.)
            raise
    except Exception as e:
        print(f"Error checking {bucket}/{key}: {e}", file=sys.stderr)
        return None


def check_s3_uri_exists(args_tuple):
    """Check if an S3 URI exists."""
    s3_uri, region = args_tuple
    try:
        bucket, key = parse_s3_uri(s3_uri.strip())
        bucket = bucket.replace('robotics-manip-lbm', 'lbm-transition')
        # Create S3 client in this thread
        s3_client = boto3.client('s3', region_name=region)
        exists = check_s3_file_exists(bucket, key, s3_client)
        return s3_uri.strip(), exists
    except ValueError as e:
        return s3_uri.strip(), None
    except Exception as e:
        return s3_uri.strip(), None


def main():
    parser = argparse.ArgumentParser(
        description="Check if S3 files exist in parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_s3_files_transfer.py s3_uris.txt
  python check_s3_files_transfer.py s3_uris.txt --max-workers 10
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Path to file containing S3 URIs (one per line)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=16,
        help='Maximum number of parallel workers (default: 16)'
    )
    
    parser.add_argument(
        '--region',
        default='us-west-2',
        help='AWS region (default: us-west-2)'
    )
    
    args = parser.parse_args()
    
    # Read S3 URIs from input file
    try:
        with open(args.input_file, 'r') as f:
            all_lines = [line.strip() for line in f if line.strip()]
            # Only check every 100th line
            s3_uris = all_lines[::100]
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not s3_uris:
        print("No S3 URIs found in input file.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Checking {len(s3_uris)} S3 URIs (every 100th line) with {args.max_workers} parallel workers...")
    print(f"Total lines in file: {len(all_lines)}")
    
    # Test AWS credentials by trying to create a client
    try:
        test_client = boto3.client('s3', region_name=args.region)
        test_client.list_buckets()  # Simple test call
    except NoCredentialsError:
        print("Error: AWS credentials not found. Please configure your AWS credentials.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error testing AWS credentials: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Check S3 files in parallel using ThreadPoolExecutor
    not_found_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit initial batch of futures up to max_workers
        futures = {}
        submitted_count = 0
        
        # Submit initial batch
        for i in range(min(args.max_workers, len(s3_uris))):
            future = executor.submit(check_s3_uri_exists, (s3_uris[i], args.region))
            futures[future] = s3_uris[i]
            submitted_count += 1
        
        with tqdm(total=len(s3_uris), desc="Checking S3 files", unit="file") as pbar:
            while futures:
                # Wait for any future to complete
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                
                # Process completed futures
                for future in done:
                    uri, exists = future.result()
                    pbar.update(1)
                    
                    if exists is False:
                        print(f"NOT_FOUND: {uri}")
                        not_found_count += 1
                    elif exists is None:
                        error_count += 1
                    
                    # Remove completed future
                    del futures[future]
                
                # Submit new futures if there are more URIs to process
                while submitted_count < len(s3_uris) and len(futures) < args.max_workers:
                    future = executor.submit(check_s3_uri_exists, (s3_uris[submitted_count], args.region))
                    futures[future] = s3_uris[submitted_count]
                    submitted_count += 1
    
    # Summary
    exists_count = len(s3_uris) - not_found_count - error_count
    
    print(f"\nSummary:")
    print(f"  Exists: {exists_count}")
    print(f"  Not Found: {not_found_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total: {len(s3_uris)}")


if __name__ == "__main__":
    main()
