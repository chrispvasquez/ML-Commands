import os
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import librosa as lb
from librosa.display import specshow

# Drop axis since data is only single channel
def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    #audio = tf.expand_dims(audio, axis=-1)
    return audio, labels

def get_spectrogram(waveform):
    spectrogram = tfio.audio.spectrogram(waveform, nfft=255, window=255, stride=128)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def get_melspectrogram(waveform, sample_rate):
    mel_spectrogram = tfio.audio.spectrogram(waveform, nfft=255, window=255, stride=128)
    mel_spectrogram = tfio.audio.melscale(mel_spectrogram, rate=sample_rate, mels=128, fmin=0, fmax=8000)
    mel_spectrogram = mel_spectrogram[..., tf.newaxis]
    return mel_spectrogram

def get_melspectrogram_db(waveform, sample_rate):
    mel_spectrogram_db = tfio.audio.spectrogram(waveform, nfft=255, window=255, stride=128)
    mel_spectrogram_db = tfio.audio.melscale(mel_spectrogram_db, rate=sample_rate, mels=128, fmin=0, fmax=8000)
    mel_spectrogram_db = tfio.audio.dbscale(mel_spectrogram_db, top_db=80)
    mel_spectrogram_db = mel_spectrogram_db[..., tf.newaxis]
    return mel_spectrogram_db

def get_mfcc(waveform, sample_rate):
    mel_spectrogram = tfio.audio.spectrogram(waveform, nfft=255, window=255, stride=128)
    mel_spectrogram = tfio.audio.melscale(mel_spectrogram, rate=sample_rate, mels=128, fmin=0, fmax=8000)
      # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrogram + 1e-6)
    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :13]
    mfcc = mel_spectrogram[..., tf.newaxis]
    return mfcc

# Function to display spectrogram
def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)

# Create Spectrogram dataset from audio files
def make_melspec_ds(ds, sr):
  return ds.map(
      map_func=lambda audio,label: (get_melspectrogram(audio, sr), label),
      num_parallel_calls=tf.data.AUTOTUNE)

# Create Spectrogram dataset from audio files
def make_melspec_db(ds, sr):
  return ds.map(
      map_func=lambda audio,label: (get_melspectrogram_db(audio, sr), label),
      num_parallel_calls=tf.data.AUTOTUNE)

# Create Spectrogram dataset from audio files
def make_mfcc_ds(ds, sr):
  return ds.map(
      map_func=lambda audio,label: (get_mfcc(audio, sr), label),
      num_parallel_calls=tf.data.AUTOTUNE)

# Create Spectrogram dataset from audio files
def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)


# Define function to check if file is in WAV format
def is_wav(filename):
    '''
        Checks if files are .wav files
        Utility tool in converting wav to png files
    '''
    return filename.split('.')[-1] == 'wav'

def opus_to_wav(clips_path, save_path):
    for subdir in os.listdir(clips_path):
        word_path = os.path.join(clips_path, subdir)
        sp = os.path.join(save_path, clips_path[:len(clips_path) - 7][-2:] + "-" + subdir)
        os.makedirs(sp)
        print("Coverting OPUS to WAV for the\"" + subdir +"\" label")
        print('++++++++++++++++++++++++++++++++++')
        for recording in os.listdir(word_path):
            recording_path = os.path.join(word_path, recording)
            wav_file = os.path.join(sp, recording.rstrip(".opus") + ".wav")
            if not os.path.exists(wav_file):
                os.system("ffmpeg -i \"" + recording_path + "\" \"" + wav_file + "\"")

def trim_audio(wav_file_loc):
    y,sr=lb.load(wav_file_loc) #load the file
    trim_file, index = lb.effects.trim(y) # Remove leading and trailing silence
    return trim_file, sr