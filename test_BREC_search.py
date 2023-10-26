import os
SEED = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

for seed in SEED:
    script_base = f'python test_BREC.py --SEED={seed} --D=2 --DEVICE=0'
    print(script_base)
    os.system(script_base)
for seed in SEED:
    script_base = f'python test_BREC.py --SEED={seed} --D=3 --DEVICE=1'
    print(script_base)
    os.system(script_base)