import math
import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from augmentation import time_domain_augmentations as TDA

def augment_data(data: pd.DataFrame, data_folder: str, path_to_save: str, augmentation=None, random_sample=True, percent=0.3, index = "0"):
    if random_sample:
        data_to_augment = data.sample(n=int(len(data) * percent))
    

    for item in tqdm(data_to_augment.iloc):
        # path_to_new_file = os.path.join(path_to_save, bird, file[:-4])
       
       # try:
        y, sr = librosa.load(os.path.join(data_folder, item['filename']), sr=32000)
            
        # Repeat audio if its length is less than the window length
        n_copy = math.ceil(5 * 32000 / len(y))
        if n_copy > 1:
            y = np.concatenate([y] * n_copy)

        # Extract a fixed-length window from the begginig of a file
        start_idx = 0
        end_idx = int(start_idx + 5 * 32000)
        y = y[start_idx:end_idx]

        y = augmentation(y)

        melspec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            fmin=60, 
            fmax=16000, 
            n_mels=128, 
            n_fft=1024)
        
        melspec = librosa.power_to_db(melspec).astype(np.float32)

        path_to_new_file = os.path.join(path_to_save, item['filename'][:-4] + augmentation.__class__.__name__)
        new_item = pd.DataFrame({"primary_label": item["primary_label"], "secondary_labels": item["secondary_labels"], "type": "", "latitude": "", "longitude": "", "scientific_name": "", "common_name": "", "author": "", "lisence": "", "rating": "", "url":"",  "filename": item['filename'][:-4] + augmentation.__class__.__name__ + ".npy"}, index=[0])
        data = pd.concat([data, new_item], ignore_index = True)

        # Skip saving if file already exists
        if not os.path.exists(path_to_new_file):
            np.save(path_to_new_file, melspec)

        
    data.to_csv("augmented_labels" + index + ".csv")
    

def create_augmented_data(audio: np.array, augmentations=None, prob=0.4) -> np.array:
    pass

def create_and_save_melspec(train_folders: list[str], path_to_save: str, config, augmenation=None):
    """
    Create and save mel spectrograms for audio files in the specified folders.

    Args:
        train_folders (list[str]): List of paths to folders containing audio data.
        path_to_save (str): Path to save the mel spectrograms.
        n_fft (int): Number of FFT points.
        n_mels (int): Number of mel bands to generate.
        min_freq (int): Minimum frequency in Hz.
        max_freq (int): Maximum frequency in Hz.
        sample_rate (int): Sampling rate of audio files.
        window_in_seconds (int): Length of the window in seconds.

    Returns:
        None
    """

    for i, data_folder in enumerate(train_folders):
        species = os.listdir(data_folder)
        print('Current folder: {}'.format(i))

        for bird in tqdm(species):

            # Check if it's a new class
            is_new_class = False
            new_dir = os.path.join(path_to_save, bird)
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)
                is_new_class = True

            files = os.listdir(os.path.join(data_folder, bird))

            for file in files:
                path_to_new_file = os.path.join(path_to_save, bird, file[:-4])
               
                try:
                    y, sr = librosa.load(os.path.join(data_folder, bird, file), sr=config.sample_rate)
            
                    # Repeat audio if its length is less than the window length
                    n_copy = math.ceil(config.max_time * config.sample_rate / len(y))
                    if n_copy > 1:
                        y = np.concatenate([y] * n_copy)

                    # Extract a fixed-length window from the begginig of a file
                    start_idx = 0
                    end_idx = int(start_idx + config.max_time * config.sample_rate)
                    y = y[start_idx:end_idx]

                    melspec = librosa.feature.melspectrogram(
                        y=y, 
                        sr=sr, 
                        fmin=config.min_frequency, 
                        fmax=config.max_frequency, 
                        n_mels=config.n_mels, 
                        n_fft=config.n_fft)
                    
                    melspec = librosa.power_to_db(melspec).astype(np.float32)

                    path_to_new_file = os.path.join(path_to_save, bird, file[:-4])

                    # melspec = cv2.resize(melspec, (256, 256), interpolation=cv2.INTER_AREA)

                    # Skip saving if file already exists
                    if is_new_class or not os.path.exists(path_to_new_file):
                        np.save(path_to_new_file, melspec)
                    
                except:
                    print('Error while processing {}'.format(os.path.join(data_folder, bird, file)))
                    continue
                                                                

def get_filtered_data(this_year_metadata: str, old_metadata: list[str], to_csv: bool=True) -> pd.DataFrame:
    """
    Gets all files with metadata and filter old datasets by next rule:
    1. Remove all files with equal name, author and label.
    2. Remove all species with less than 70 elements.

    Concatenate all data together
    """
    old_data = pd.DataFrame()
    for file in old_metadata:
        old_data = pd.concat([old_data, pd.read_csv(file)])

    print('Total number of elements: {}'.format(len(old_data)))
    old_data = old_data.drop_duplicates(['filename', 'primary_label', 'author'])
    old_data = old_data.groupby('primary_label').filter(lambda group: len(group) >= 70)
    print('Number of elements in filtered data: {}'.format(len(old_data)))

    new_data = pd.read_csv(this_year_metadata)
    data = pd.concat([new_data, old_data])

    if to_csv:
        new_data.to_csv('this_year_only_metadata.csv', index=False)
        old_data.to_csv('past_years_metadata.csv', index=False)
        data.to_csv('full_dataset_metadata.csv', index=False)

    return data

def get_species_with_low_elements(data: pd.DataFrame) -> list[str]:
    """
    """

    return data.groupby('primary_label').filter(lambda group: len(group) < 70)['primary_label'].unique().tolist()

def fill_wrong_filepath(data_file):
    def correct_path(row):
        label = row['primary_label']
        path = row['filename']
        if not path.startswith(label):
            path = f"{label}/{path}"
        return path

    data = pd.read_csv(data_file)
    data['filename'] = data.apply(correct_path, axis=1)
    data.to_csv(data_file)
    

if __name__ == '__main__':

    i = 0

    data = pd.read_csv('/workspace/datasets/birdclef_datasets/bc_2024/train_metadata.csv')

    birds = data['primary_label'].unique()

    for i in birds:
        os.mkdir(os.path.join( '/workspace/birdclef/augmented_melspecs', i))

    augs = [
        TDA.Echo(0.2, 0.5, 32000), 
        TDA.Reverb('/workspace/birdclef/ir_respone.npy'), 
        TDA.RandomVolumeChange(), 
        TDA.WhiteNoise()
    ]

    for aug, prob, index in zip(augs, [0.3, 0.4, 0.1, 0.4], ["0", "1", "2", "3"]): 
        augment_data(data,
                     '/workspace/datasets/birdclef_datasets/bc_2024/train_audio', 
                     '/workspace/birdclef/augmented_melspecs', 
                     aug, True, prob, index)

    
    




    
    

