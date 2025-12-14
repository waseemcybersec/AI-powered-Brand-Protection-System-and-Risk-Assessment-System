#!/usr/bin/env python3
"""
DNS Check: Filter live domains from candidates.txt
"""
import socket
import threading
from queue import Queue
from pathlib import Path

INPUT_FILE = "candidates.txt"
OUTPUT_FILE = "live_domains.txt"
NUM_THREADS = 50  # adjust for your CPU/network

live_domains = set()
queue = Queue()

# Worker function
def worker():
    while True:
        domain = queue.get()
        if domain is None:
            break
        try:
            socket.gethostbyname(domain)
            live_domains.add(domain)
            print(f"[LIVE] {domain}")
        except socket.gaierror:
            pass  # domain doesn't exist
        except Exception as e:
            print(f"[WARNING] Error checking {domain}: {e}")
        finally:
            queue.task_done()

def main():
    # Get proper file paths
    script_dir = Path(__file__).parent.absolute()
    input_path = script_dir / INPUT_FILE
    output_path = script_dir / OUTPUT_FILE
    
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return
    
    # Read all candidates
    with open(input_path, 'r', encoding='utf-8') as f:
        domains = [line.strip() for line in f if line.strip()]

    print(f"[INFO] Checking {len(domains)} domains with {NUM_THREADS} threads...")

    # Start threads
    threads = []
    for _ in range(NUM_THREADS):
        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()
        threads.append(t)

    # Enqueue domains
    for d in domains:
        queue.put(d)

    # Wait for all tasks to finish
    queue.join()

    # Stop threads
    for _ in threads:
        queue.put(None)
    for t in threads:
        t.join()

    # Save live domains
    with open(output_path, 'w', encoding='utf-8') as f:
        for d in sorted(live_domains):
            f.write(d + "\n")

    print(f"[✔] Live domains saved: {output_path} ({len(live_domains)}/{len(domains)})")

if __name__ == "__main__":
    main()

