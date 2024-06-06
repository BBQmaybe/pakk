import os
import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import cv2

import toml
from augmentation import frequency_domain_augmentations as FDA


class DatasetUtils:
    @staticmethod
    def make_melspectogram(audio_data, current_config):
        melspec = librosa.feature.melspectrogram(
            y=audio_data, sr=current_config['data_parameters']['sample_rate'], n_mels=current_config['spectrogram_parameters']['n_mels'], 
            fmin=current_config['data_parameters']['min_frequency'], fmax=current_config['data_parameters']['max_frequency'],
        )
        return librosa.power_to_db(melspec).astype(np.float32)
    
    @staticmethod
    def convert_to_colored_image(data, eps=1e-6):
        mean = data.mean()
        std = data.std()

        data = (data - mean) / (std + eps)
        _min, _max = data.min(), data.max()

        if (_max - _min) > eps:
            image = np.clip(data, _min, _max)
            image = 255 * (image - _min) / (_max - _min)
            # image = image.astype(np.uint8)
        else:
            image = np.zeros_like(data, dtype=np.uint8)
            
        return image
    
    @staticmethod
    def prepare_data(row: pd.DataFrame, current_config) -> tuple[torch.Tensor, torch.Tensor]:
        path = os.path.join(current_config['meta_parameters']['path_to_data'], row['filename'])
        path = path.split('.')[0]
        path += '.npy'

        mel_spectogram = np.load(path)
#        mel_spectogram = cv2.resize(mel_spectogram, (256, 256), interpolation=cv2.INTER_AREA)
        labels = torch.tensor(list(row)[2:]).float()

        return mel_spectogram, labels


class BirdCLEFDataset(Dataset):
    def __init__(self, data, current_config, augmentations=None):
        super().__init__()
        self.data = data
        self.current_config = current_config
        self.augmentations = augmentations 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data, labels = DatasetUtils.prepare_data(self.data.loc[idx], self.current_config)
        data = DatasetUtils.convert_to_colored_image(data)

        if self.augmentations:
            data = self.augmentations(image=data)['image']
       
            # # Use mix up here
            # if np.random.uniform(0, 1) < 0.35:
            #     random_sample_idx = np.random.randint(0, len(self.data))
            #     new_data, new_label = DatasetUtils.prepare_data(self.data.loc[random_sample_idx], self.current_config)
            #     new_data = DatasetUtils.convert_to_colored_image(data)
            #     new_data = self.augmentations(image=new_data)['image']
            #     data, labels = FDA.mix_up(data, labels, new_data, new_label)

        return torch.tensor(data, dtype=torch.float32), labels
