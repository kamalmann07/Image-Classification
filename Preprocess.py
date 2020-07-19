import cv2
import numpy as np
import os, glob

path = ['C:/Users/Kamal/PycharmProjects/ML Project/training/n0', 'C:/Users/Kamal/PycharmProjects/ML Project/training/n1', 'C:/Users/Kamal/PycharmProjects/ML Project/training/n2', 'C:/Users/Kamal/PycharmProjects/ML Project/training/n3', 'C:/Users/Kamal/PycharmProjects/ML Project/training/n4', 'C:/Users/Kamal/PycharmProjects/ML Project/training/n5', 'C:/Users/Kamal/PycharmProjects/ML Project/training/n6', 'C:/Users/Kamal/PycharmProjects/ML Project/training/n7', 'C:/Users/Kamal/PycharmProjects/ML Project/training/n8', 'C:/Users/Kamal/PycharmProjects/ML Project/training/n9']

extension = 'jpg'
os.chdir(path[0])
inboundFiles = [i for i in glob.glob('*.{}'.format(extension))]

# add light color
def add_light_color(image, color, gamma=1.0):
  file_name = image.split('.')[0]
  image = cv2.imread(image)
  invGamma = 1.0 / gamma
  image = (color - image)
  table = np.array([((i / 255.0) ** invGamma) * 255
                    for i in np.arange(0, 256)]).astype("uint8")

  image = cv2.LUT(image, table)
  gamma = 1.0
  cv2.imwrite(file_name + '_alc' + '.jpg', image)

# flip image
def flip_image(image_name):
    file_name = image_name.split('.')[0]
    image = cv2.imread(image_name)
    image = cv2.flip(image, 0)
    cv2.imwrite(file_name+'_flip.jpg', image)

# brigtness
def add_light(image, gamma=1.50):
  file_name = image.split('.')[0]
  image = cv2.imread(image)
  invGamma = 1.0 / gamma
  table = np.array([((i / 255.0) ** invGamma) * 255
                    for i in np.arange(0, 256)]).astype("uint8")

  image = cv2.LUT(image, table)
  cv2.imwrite(file_name + '_light.jpg', image)

# for file in inboundFiles:
#     img = cv2.imread(file)
#     cv2.imshow('image', img)
#     cv2.waitKey(0)

for i, image in enumerate(inboundFiles):
    # Logic used to randomly modify images to make dataset diverse
    if i % 5 == 0:
        add_light_color(image, 150)
    if i % 8 == 0:
        add_light(image)
