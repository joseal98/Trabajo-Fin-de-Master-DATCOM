import math, random
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio


class AudioUtil():
  # -------------------------------------------------------------------------
  # Lectura del audio. Devuelve la señal como tensor y la tasa de muestreo
  # -------------------------------------------------------------------------
  @staticmethod
  def open(audio_file):
    sig, sr = torchaudio.load(audio_file)
    return (sig, sr)

  # -------------------------------------------------------------------------
  # Guardar el audio en un archivo. Devuelve True si todo va bien.
  # -------------------------------------------------------------------------
  @staticmethod
  def save(audio, dest_path):
    try:
      sig, sr = audio
      torchaudio.save(dest_path, sig, sr)
      return True
    except:
      return False

  # -----------------------------------------------------------
  # Limpieza de los primeros y últimos instantes del audio
  # -----------------------------------------------------------
  @staticmethod
  def prune_audio(aud, start=20000, end=-20000):
    sig, sr = aud
    return (sig[:,start:end], sr)

  # -------------------------------------------------------------------------
  # Fragmentación del audio en trozos de igual duracion (segundos)
  # Si no se especifica la duracion, se usa num (numero de trozos)
  # -------------------------------------------------------------------------
  @staticmethod
  def split_audio(aud, duration=None, num=10):
    sig, sr = aud
    audio_list = []

    if duration:
      duration *= sr
      num = math.floor(sig.shape[1]/(duration))
      rem = (num+1)*duration - sig.shape[1]
      for i in range(num):
        audio_list.append((sig[:,i*duration:(i+1)*duration], sr))
      audio_list.append( ( torch.cat( [sig[:,(i+1)*duration:], sig[:,:rem]], dim=1 ), sr) )
    else:
      duration = math.floor(sig.shape[1]/num)
      for i in range(num):
        audio_list.append((sig[:,i*duration:(i+1)*duration], sr))

    return audio_list

  # ------------------------------------------------------
  # Representacion de la onda sonora. Tiempo-Amplitud
  # ------------------------------------------------------
  @staticmethod
  def plot_waveform(aud, title=None, xlim=None, ylim=None):
    waveform, sample_rate = aud
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
      axes = [axes]
    for c in range(num_channels):
      axes[c].plot(time_axis, waveform[c], linewidth=1)
      axes[c].grid(False)
      if num_channels > 1:
        axes[c].set_ylabel(f'Channel {c+1}')
      if xlim:
        axes[c].set_xlim(xlim)
      if ylim:
        axes[c].set_ylim(ylim)
    if title:
      figure.suptitle(title)
    plt.show(block=False)

  # ----------------------------
  # Cambiar numero de canales
  # ----------------------------
  @staticmethod
  def rechannel(aud, new_channel):
    sig, sr = aud

    if (sig.shape[0] == new_channel):
      # Nada que hacer
      return aud

    if (new_channel == 1):
      # Conversion de estereo a mono tomando solo el primer canal
      resig = sig[:1, :]
    else:
      # Conversion de mono a estereo duplicando el canal existente
      resig = torch.cat([sig, sig])

    return ((resig, sr))


  # ----------------------------
  # Resampleado
  # ----------------------------
  @staticmethod
  def resample(aud, newsr):
    sig, sr = aud

    if (sr == newsr):
      # Nothing to do
      return aud

    num_channels = sig.shape[0]
    # Resample first channel
    resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
    if (num_channels > 1):
      # Resample the second channel and merge both channels
      retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
      resig = torch.cat([resig, retwo])

    return ((resig, newsr))

  # ----------------------------
  # Generacion del espectrograma
  # ----------------------------
  @staticmethod
  def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
    sig,sr = aud
    top_db = 80

    # spec tiene dimensiones [channel, n_mels, time], done channel es mono, stereo, etc
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # Convertir a decibelios
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec)


  # -----------------------------------------------------------------------------
  # Aumento de datos por medio de enmascarar franjas horizontales y/o verticales
  # Las máscaras aplicadas se corresponden con el valor medio
  # -----------------------------------------------------------------------------
  @staticmethod
  def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
      aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
      aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

    return aug_spec


class AudioAugmentation():

  # ---------------------------------------------------------------------------
  # Truncar o completar la señal hasta una longitud maxima 'max_s' en segundos
  # ---------------------------------------------------------------------------
  @staticmethod
  def pad_trunc(aud, max_s):
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr * max_s

    if (sig_len > max_len):
      # Truncar la señal a una duracion determinada
      sig = sig[:,:max_len]

    elif (sig_len < max_len):
      pad_begin_len = random.randint(0, max_len - sig_len)
      pad_end_len = max_len - sig_len - pad_begin_len

      # Se rellena con 0s
      pad_begin = torch.zeros((num_rows, pad_begin_len))
      pad_end = torch.zeros((num_rows, pad_end_len))

      sig = torch.cat((pad_begin, sig, pad_end), 1)

    return (sig, sr)

  # --------------------------------------------------------
  # Traslacion del audio por un porcentaje de su duracion
  # --------------------------------------------------------
  @staticmethod
  def time_shift(aud, shift_limit=0.7):
    sig,sr = aud
    _, sig_len = sig.shape
    shift_amt = int(random.random() * shift_limit * sig_len)
    return (sig.roll(shift_amt), sr)

  # ----------------------------
  # Adicion de ruido
  # ----------------------------
  @staticmethod
  def add_noise(aud, fctr=0.05):
    sig,sr = aud
    noise = torch.randn(sig.shape)
    data_noise = sig + fctr * noise
    return (data_noise, sr)