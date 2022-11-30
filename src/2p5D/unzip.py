from sys import argv
import os
from glob import glob

path = argv[1]

print(path)

os.chdir(path)

files = glob("*.zip")

try:
    os.mkdir("unzipped")
except:
    pass

for file in files:
    os.system(f'unzip {file} -d unzipped/')


