import os
import sys
path = sys.argv[1]
n_validation = 20

for i in range(13):
    os.system(f'mkdir -p {path}/validate/{i}/')

for i in range(13):
    cnt = 0
    for filename in os.listdir(f'{path}/train/{i}/'):
        if not os.path.isfile(f'{path}/train/{i}/{filename}'):
            continue
        if cnt > n_validation:
            break
        cnt += 1
        print(f'mv {path}/train/{i}/{filename} {path}/validate/{i}/')
        os.system(f'mv {path}/train/{i}/{filename} {path}/validate/{i}/')
        
