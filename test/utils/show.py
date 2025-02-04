# In current directory, each folder has a file named "xx/GW_Hamiltonian_matrix.png", this script will show them one by one.
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

def show_image(image_path):
    img = mpimg.imread(image_path)
    plt.figure(dpi=300)  # Increase the resolution
    imgplot = plt.imshow(img)
    plt.show()

def main(path="."):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            if os.path.exists(name+"/GW_Hamiltonian_matrix.png"):
                print(name)
                show_image(name+"/GW_Hamiltonian_matrix.png")
                time.sleep(1)
                

if __name__ == "__main__":
    main(".")