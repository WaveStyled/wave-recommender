import requests


if __name__ == '__main__':
    response = requests.put('http://127.0.0.1:5001/wardrdrobepython/')
    print(response.json())