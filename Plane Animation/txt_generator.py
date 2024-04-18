import time
import os


plane = '''
/-----------------------\                                     
|                       |                    |~~\_____/~~\__  
|      Midiendo...      |____________________ \______====== )-
|                       |                            ~~~|/~~  
\-----------------------/                               ()    
'''

propellers = ['\n  \n  \n+ \n  \n  \n', '\n  \n| \n+ \n| \n  \n']


def multi_concatenate(str1, str2):
    
    split1 = str1.split('\n')
    split2 = str2.split('\n')
    
    multi_string = '\n'.join(''.join(line) for line in zip(split1, split2))
    
    return multi_string


def multi_slice(str, start, end):
    
    split = str.split('\n')
    
    sliced_str = '\n'.join(''.join(line[start:end]) for line in split)
    
    return sliced_str


len_plane = 64

os.system('cls')  # Clear console

# Animation starts

with open('./plane.txt', 'w') as planetxt:

    for i in range(1, len_plane+1): # Plane appears
        
        full_plane = multi_concatenate(plane, propellers[i%2])
        
        print(multi_slice(full_plane, -i, -1))
        print('\033[7A\033[2K', end='')
        
        time.sleep(0.1)
        
        planetxt.write(multi_slice(full_plane, -i, -1))
        planetxt.write('END')
        
    

    n_spaces = 1

    for i in range(77):  # Plane flies
        
        full_plane = multi_concatenate(plane, propellers[i%2])
        air = f"{' '*n_spaces}\n"*6
        full_plane = multi_concatenate(air, full_plane)
        
        print(multi_slice(full_plane, 0, -1))
        print('\033[7A\033[2K', end='')
        time.sleep(0.1)
        
        n_spaces += 1
        
        planetxt.write(multi_slice(full_plane, 0, -1))
        planetxt.write('END')
    
    for i in range(2, len_plane+1):  # Plane dissapears
        
        full_plane = multi_concatenate(plane, propellers[i%2])
        air = f"{' '*n_spaces}\n"*6
        full_plane = multi_concatenate(air, full_plane)
        
        print(multi_slice(full_plane, 0, -i))
        print('\033[7A\033[2K', end='')
        time.sleep(0.1)
        
        n_spaces += 1
        
        planetxt.write(multi_slice(full_plane, 0, -i))
        if i != len_plane:
            planetxt.write('END')