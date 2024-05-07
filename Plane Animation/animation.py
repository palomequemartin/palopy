import time
import os

os.system('cls')

with open('./plane.txt', 'r') as planetxt:
    animation = planetxt.read().split('END')
    while True:
        time.sleep(1)
        for frame in animation:
            print(frame)
            print('\033[7A\033[2K', end='')
            time.sleep(0.1)