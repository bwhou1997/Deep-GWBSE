import json
import os

with open("flows-nc300/metal_seek.json", "r") as file:
    metal_seek_data = json.load(file)
    metal_list = metal_seek_data["metal"]
    
    for folder in metal_list:
        print(folder)
        os.system(f"cp -r {folder} metal/")
