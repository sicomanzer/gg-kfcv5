from curl_cffi import requests
import pandas as pd
from bs4 import BeautifulSoup
import json
import os

# --- SNIPER SCRAPING DEMO ---
def get_major_shareholders(symbol):
    """
    Fetches major shareholders from SET Website (Unofficial Sniper Method).
    Target: https://www.set.or.th/api/set/factsheet/{symbol}/major-shareholders?lang=th
    Using curl_cffi to bypass 403 Forbidden (Cloudflare/WAF).
    """
    if not symbol: return None
    
    # Clean symbol (remove .BK if present)
    clean_symbol = symbol.replace('.BK', '').upper()
    
    url = f"https://www.set.or.th/api/set/factsheet/{clean_symbol}/major-shareholders?lang=th"
    
    # Minimal headers often work better with impersonate
    headers = {
        'Referer': 'https://www.set.or.th/',
        'Accept': 'application/json'
    }
    
    try:
        # impersonate="chrome110" helps bypass TLS fingerprint checks
        response = requests.get(url, headers=headers, impersonate="chrome110", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'majorShareholders' in data and data['majorShareholders']:
                # Extract relevant fields
                holders = []
                for item in data['majorShareholders']:
                    holders.append({
                        'Name': item.get('name', '-'),
                        'Shares': item.get('share', 0),
                        'Percent': item.get('pctShare', 0.0)
                    })
                return pd.DataFrame(holders)
            else:
                return None
        else:
            print(f"Failed to scrape {clean_symbol}: Status {response.status_code}")
            return None
    except Exception as e:
        print(f"Scraping Error for {clean_symbol}: {e}")
        return None

def send_line_notify(token, message):
    url = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {token}'}
    data = {'message': message}
    try:
        response = requests.post(url, headers=headers, data=data)
        return response.status_code == 200, response.text
    except Exception as e:
        return False, str(e)

CONFIG_FILE = "config.json"

def load_config():
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)
