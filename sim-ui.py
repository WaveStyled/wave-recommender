import requests

def main_loop():
    while True:
        x = input()
        if(x == "calibrate"):
            return
        if(x == "recommend"):
            return

# Calls server startup, loads the wardrobe csv
def bootup():
    # call bootup link
    r = requests.put("http://localhost:5001/start")

def calibrate_start():
    print("How many calibration outfits would you like to see?\n")
    print("Note: More you calibrate, more the model will understand your likes:\n")
    num_calibrate = int(input())
    r = requests.put("http://localhost:5001/calibrate_start/"+str(num_calibrate))
    fits = r.json()
bootup()
calibrate_start()


#main_loop()