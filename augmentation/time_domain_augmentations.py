import numpy as np
import librosa
from scipy.signal import convolve

def random_apply(x, augmentation, prob):
    if np.random.uniform(0, 1) < prob:
        return augmentation(x)
    else:
        return x

class WhiteNoise:
  def __init__(self):
    pass
      
  def __call__(self, x):
    noised_signal = x + np.random.normal(0, 0.0005, x.shape)
    return noised_signal


class RandomVolumeChange:
  def __init__(
      self, 
      high_limit: int=10, 
      low_limit: int=-10) -> None:
    
    self.high_limit = high_limit
    self.low_limit = low_limit

  @staticmethod
  def chahge_volume(x, db):
    return x * RandomVolumeChange._db2float(db)
  
  @staticmethod
  def volume_down(x, db):
    return x * RandomVolumeChange._db2float(db)
    
  def __call__(self, x):
    db = self._db2float(
        np.random.uniform(self.low_limit, self.high_limit)
      )
    
    return self.chahge_volume(x, db)

  @staticmethod
  def _db2float(db: float):
    return 10 ** (db / 20)
  
  
class TimeStretch:
  def __init__(self, sample_rate=32000):
      self.sample_rate = sample_rate

  def __call__(self, input) -> np.array:
      rate = np.random.uniform(0, 1)
      augmented = librosa.effects.time_stretch(y=input,  rate=rate)
      return librosa.util.normalize(augmented)
  

def recover_volume(signal, augmented_signal)  -> np.array:
  rms_original = np.sqrt(np.mean(signal**2))
  rms_reverb = np.sqrt(np.mean(augmented_signal**2))
  return augmented_signal * (rms_original / rms_reverb) 


class Echo:
  def __init__(self, delay, decay, sample_rate):
    self.delay = int(delay * sample_rate)
    self.decay = decay

  def __call__(self, x):
    num_samples = len(x)
    echo_array = np.zeros(num_samples, dtype=np.float32)
    echo_array[0:num_samples] += x
    echo_array[self.delay:num_samples] += self.decay * x[0:num_samples - self.delay]
    return recover_volume(x, echo_array)
  

class Reverb:
  def __init__(self, ir_resonse_file: str):
    self.ir_array  = np.load(ir_resonse_file)

  def  __call__(self, x):
    rms_original = np.sqrt(np.mean(x**2))
    reverb_audio = convolve(x, self.ir_array, mode='full')[:len(x)]
    rms_reverb = np.sqrt(np.mean(reverb_audio**2))
    reverb_audio = reverb_audio * (rms_original / rms_reverb)
    return recover_volume(x, reverb_audio)