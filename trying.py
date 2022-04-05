import requests  ## DUMMY TESTER CLASS

response = requests.get(url="http://localhost:5001/wardrobedata")
print(response.text)