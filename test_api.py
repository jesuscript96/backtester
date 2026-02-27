import urllib.request
import urllib.error

endpoints = ["http://localhost:8000/api/datasets", "http://localhost:8000/api/strategies"]

for url in endpoints:
    print(f"--- Fetching {url} ---")
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            print("SUCCESS:", response.read().decode())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()}")
    except Exception as e:
        print(f"Error: {e}")
