import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

os.environ['CPP_TF_MIN_LOG_LEVEL'] = '2'

# folder path
raw_data_path = './datasets/raw_data'
positive_path = './datasets/positive/'
negative_path = './datasets/negative/'
# image dataframe path
image_path_df = './datasets/image_paths.csv'
positive_df_path = './datasets/pos.csv'
negative_df_path = './datasets/neg.csv'

# image height and width after jitter
IMG_HEIGHT, IMG_WIDTH = 105, 105

def load_image_and_labels():
    # image labels start from 0
    labels = 0
    image_paths = []
    image_labels = []

    ### loading all the image path and it's categorical into arrays
    ### looping by alphabet
    for alphabet in os.listdir(raw_data_path):
        alphabet_path = os.path.join(raw_data_path, alphabet)
        # looping by character
        for character in os.listdir(alphabet_path):
            character_path = os.path.join(alphabet_path, character)
            # looping image one by one
            for image in os.listdir(character_path):
                image_paths.append(os.path.join(character_path, image))
                image_labels.append(labels)
            labels += 1

    return image_paths, image_labels


def generate_positive_samples():
    if not os.path.isfile(positive_df_path):
        df = pd.read_csv(image_path_df)
        df.loc[:, 'image2'] = df['image'].shift(-1)
        df.rename(columns={'image':'image1'}, inplace=True)
        df = df.groupby('label').head(10).reset_index()
        df.drop(columns=['index'], inplace=True)
        df = df[['image1','image2','label']]
        df.to_csv(positive_df_path, index=False)


def generate_negative_samples():
    if not os.path.isfile(negative_df_path):
        df = pd.read_csv(image_path_df)
        df_shuffle = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df.loc[:, 'image2'] = df_shuffle['image']
        df.loc[:, 'label2'] = df_shuffle['label']
        # select only not matched records
        df = df.loc[~(df['label'] == df['label2'])]
        # only select top 10 for each category
        df = df.groupby('label').head(10).reset_index()
        df.rename(columns={'image':'image1'}, inplace=True)
        df = df[['image1','image2','label']]
        df.to_csv(negative_df_path, index=False)


def resize_image(image1, image2, height, width):
    image1 = tf.image.resize(image1, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image2 = tf.image.resize(image2, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image1, image2


def random_crop(image1, image2):
    # [150, 150, 3] => [2, 150, 150, 3]
    stacked_images = tf.stack([image1, image2], axis=0)
    cropped_images = tf.image.random_crop(stacked_images, [2, IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_images[0], cropped_images[1]


@tf.function
def random_jitter(image1, image2):
    # resize image to 150 * 150
    image1, image2 = resize_image(image1, image2, 130, 130)
    # random crop image
    image1, image2 = random_crop(image1, image2)
    # resize to a smaller img for better local training
    image1, image2 = resize_image(image1, image2, 32, 32)
    # flip image
    if tf.random.uniform(()) > 0.5:
        image1 = tf.image.flip_left_right(image1)
        image2 = tf.image.flip_left_right(image2)

    return image1, image2



def concat_images_and_write_target_folder(target_file, target_folder):
    # if image already generated
    if len(os.listdir(target_folder)) == 16230:
        return
    samples = pd.read_csv(target_file)
    idx = 1
    for path1, path2 in zip(samples['image1'].values, samples['image2']):
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        img1, img2 = random_jitter(img1, img2)
        concat_img = np.concatenate((img1, img2), axis=0)
        cv2.imwrite(os.path.join(target_folder, f'{idx}.jpg'), concat_img)
        idx += 1


if __name__ == '__main__':

    positive_df = pd.DataFrame()
    negative_df = pd.DataFrame()

    ### loading image path and labels, save into a csv file
    if not os.path.isfile(image_path_df):

        image_paths, image_labels = load_image_and_labels()

        image_file_df = pd.DataFrame(data={'image':image_paths, 'label':image_labels})

        image_file_df.to_csv(image_path_df, index=False)

    ### generate positive samples and negative samples
    ### because every character folder contains 20 images which belongs to same category
    ### so for each character folder we're going to generate 10 positive samples and 10 negative samples
    ### here positive samples means image from the same character folder
    ### while negative samples means image from different character folder
    generate_positive_samples()
    generate_negative_samples()
    concat_images_and_write_target_folder(positive_df_path, positive_path)
    concat_images_and_write_target_folder(negative_df_path, negative_path)


















