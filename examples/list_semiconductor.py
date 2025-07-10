import json
import os

with open("flows-nc300/metal_seek.json", "r") as file:
    metal_seek_data = json.load(file)
    semiconductor_list = metal_seek_data["semiconductor"]
    
    for folder in semiconductor_list:
        print(folder)
        os.system(f"cp -r nc300/{folder} nc300-semiconductor/")
