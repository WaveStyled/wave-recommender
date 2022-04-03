import requests


if __name__ == '__main__':
    response = requests.put('http://localhost:5001/ping/', json={'update':1})
    print(response)