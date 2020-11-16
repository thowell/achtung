import imageio
import os
import sys

for i in [2, 4, 9, 15, 29, 30, 60, 64, 73, 84, 90, 91, 92, 94]:
    images = []
    m = len(os.listdir("images/game_{}".format(i)))
    for k in range(1,m+1):
        _filename = "images/game_{}/{}.JPG".format(i,k)
        # print(_filename)
        images.append(imageio.imread(_filename))
    imageio.mimsave('gifs/game_{}_{}.gif'.format(i, m), images, duration = 0.05)
    print("gif for game", i)
