API_KEY = "IiF****55m735****G"
API_SECRET = "nV****hR65TTKh71L****6dZWyU7YjWxdXlb"

exchange_config = {
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'adjustForTimeDifference': True,
        'recvWindow': 10000
    },
    'timeout': 30000
}
