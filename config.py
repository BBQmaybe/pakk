import pandas as pd
import toml

def get_labels_from_metadata(metadata_path):
    return list(pd.read_csv(metadata_path)['primary_label'].unique())

def create_default_config():
    config = {
        "spectrogram_parameters": {
            "n_fft": 1024,
            "num_fold": 5,
            "window_duration_in_sec": 5,
            "n_mels": 128
        },
        "data_parameters": {
            "max_time": 5,
            "sample_rate": 32000,
            "audio_length": 5 * 32000,
            "min_frequency": 60,
            "max_frequency": 16000
        },
        "augmentation_probabilities": {
            "WhiteNoise": 0.3, 
            "RandomVolumeChange": 0.1,
            "Reverb": 0.4,
            "Echo": 0.3,
        },
        "model_parameters": {
            "model_name" : "efficientnet_b2",
            "batch_size": 32,
            "inference_chunks_number": 48,
            "epochs": 30,
            "learning_rate": 6e-4,
            "num_classes": 182
        },
        "optimizer_parameters": {
            "eta_min": 1e-6
        },
        "meta_parameters": {
            "path_to_data": "/workspace/birdclef/melspecs",
            "metadata": "/workspace/birdclef/labels/this_year_only_metadata.csv"
        }
    }
    return config

def save_config(config, filename):
    with open(filename, 'w') as f:
        toml.dump(config, f)

def load_config(filename):
    with open(filename, 'r') as f:
        config = toml.load(f)
    # config['paths']['labels'] = get_labels_from_metadata(config['paths']['metadata'])
    return config