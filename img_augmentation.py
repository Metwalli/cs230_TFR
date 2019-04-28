# Image preprocessing in Keras

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import random
from imutils import paths
from tqdm import tqdm
import argparse
datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        # horizontal_flip=True,
        fill_mode='nearest')

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',
                    help="Data Directory")
parser.add_argument('--num_aug',
                    help="Number of Augmentations"),

args = parser.parse_args()
data_dir = args.data_dir
no_aug = int(args.num_aug)

def img_augmentation(img, output_dir, filename):
    x = img_to_array(img)  # convert image to numpy array
    x = x.reshape((1,) + x.shape)  # reshape image to (1, ..,..,..) to fit keras' standard shape

    # Use flow() to apply data augmentation randomly according to the datagenerator
    # and saves the results to the `preview/` directory
    num_image_generated = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix=filename, save_format='jpg'):

        num_image_generated += 1
        if num_image_generated == no_aug:
            break # stop the loop after num_image_generated iterations



imagePaths = sorted(list(paths.list_images(data_dir)))
labels = []
print("Processing {} data, saving preprocessed data to")
for filename in tqdm(imagePaths):
    class_name = filename.split(os.path.sep)[-2]
    prefix = filename.split(os.path.sep)[-1].split('.')[0]
    # print(class_name)
    img = load_img(filename)
    img_augmentation(img, os.path.join(data_dir, class_name), prefix)
