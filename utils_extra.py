
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
