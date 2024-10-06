# Google Colabでこのチュートリアルを実行するには、
# 次のコマンドで必要なパッケージをインストールします
# !pip install torchaudio librosa boto3

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

print(torch.__version__)
print(torchaudio.__version__)


# SAMPLE_WAV_PATH = "/Users/ken/Desktop/kaggle_ws/src/Audio_data/_sample_data/speech.wav"
import os
import io
import os
import math
import tarfile
import multiprocessing

import scipy
import librosa
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import requests
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Audio, display
# _SAMPLE_DIR = "_sample_data"
# SAMPLE_WAV_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/steam-train-whistle-daniel_simon.wav"
# SAMPLE_WAV_PATH = os.path.join(_SAMPLE_DIR, "steam.wav")
_SAMPLE_DIR = "_sample_data"
SAMPLE_WAV_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/steam-train-whistle-daniel_simon.wav"
SAMPLE_WAV_PATH = os.path.join(_SAMPLE_DIR, "steam.wav")

SAMPLE_WAV_SPEECH_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
SAMPLE_WAV_SPEECH_PATH = os.path.join(_SAMPLE_DIR, "speech.wav")

SAMPLE_RIR_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/distant-16k/room-response/rm1/impulse/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo.wav"
SAMPLE_RIR_PATH = os.path.join(_SAMPLE_DIR, "rir.wav")

SAMPLE_NOISE_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/distant-16k/distractors/rm1/babb/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo.wav"
SAMPLE_NOISE_PATH = os.path.join(_SAMPLE_DIR, "bg.wav")

SAMPLE_MP3_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/steam-train-whistle-daniel_simon.mp3"
SAMPLE_MP3_PATH = os.path.join(_SAMPLE_DIR, "steam.mp3")

SAMPLE_GSM_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/steam-train-whistle-daniel_simon.gsm"
SAMPLE_GSM_PATH = os.path.join(_SAMPLE_DIR, "steam.gsm")

SAMPLE_TAR_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit.tar.gz"
SAMPLE_TAR_PATH = os.path.join(_SAMPLE_DIR, "sample.tar.gz")
SAMPLE_TAR_ITEM = "VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"

def print_metadata(metadata, src=None):
  if src:
    print("-" * 10)
    print("Source:", src)
    print("-" * 10)
  print(" - sample_rate:", metadata.sample_rate)
  print(" - num_channels:", metadata.num_channels)
  print(" - num_frames:", metadata.num_frames)
  print(" - bits_per_sample:", metadata.bits_per_sample)
  print(" - encoding:", metadata.encoding)
  print()

def print_stats(waveform, sample_rate=None, src=None):
  if src:
    print("-" * 10)
    print("Source:", src)
    print("-" * 10)
  if sample_rate:
    print("Sample Rate:", sample_rate)
  print("Shape:", tuple(waveform.shape))
  print("Dtype:", waveform.dtype)
  print(f" - Max:     {waveform.max().item():6.3f}")
  print(f" - Min:     {waveform.min().item():6.3f}")
  print(f" - Mean:    {waveform.mean().item():6.3f}")
  print(f" - Std Dev: {waveform.std().item():6.3f}")
  print()
  print(waveform)
  print()

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  plt.show(block=False)

def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].specgram(waveform[c], Fs=sample_rate)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
  figure.suptitle(title)
  plt.show(block=False)

def play_audio(waveform, sample_rate):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  if num_channels == 1:
    display(Audio(waveform[0], rate=sample_rate))
  elif num_channels == 2:
    display(Audio((waveform[0], waveform[1]), rate=sample_rate))
  else:
    raise ValueError("Waveform with more than 2 channels are not supported.")


metadata = torchaudio.info(SAMPLE_WAV_PATH)
print_metadata(metadata, src=SAMPLE_WAV_PATH)

# 日本語訳注
# 上記で使用しているSAMPLE_WAV_PATHは、
# セル「データとユーティリティ関数の準備
# このセルの中身を確認する必要はありません。
# 本セルを一度実行して、次のセルへと進んでください」を
# 実行した際に作られる変数です

metadata = torchaudio.info(SAMPLE_MP3_PATH)
print_metadata(metadata, src=SAMPLE_MP3_PATH)

metadata = torchaudio.info(SAMPLE_GSM_PATH)
print_metadata(metadata, src=SAMPLE_GSM_PATH)

"""
# with requests.get(SAMPLE_WAV_URL, stream=True) as response:
#   metadata = torchaudio.info(response.raw)
# print_metadata(metadata, src=SAMPLE_WAV_URL)

# with requests.get(SAMPLE_MP3_URL, stream=True) as response:
#   metadata = torchaudio.info(response.raw, format="mp3")

#   print(f"Fetched {response.raw.tell()} bytes.")
# print_metadata(metadata, src=SAMPLE_MP3_URL)

# 日本語訳注
# SAMPLE_MP3_URLをファイルパスのオブジェクトとしてinfoに渡した3つほど上のセルでは
# きちんと情報が取得できていますが、パスからファイルを取ってきてファイルのオブジェクトで
# 渡した本セルでは、num_framesなどが0になりきちんと、読み取れていません。
"""

waveform, sample_rate = torchaudio.load(SAMPLE_WAV_SPEECH_PATH)

print_stats(waveform, sample_rate=sample_rate)
plot_waveform(waveform, sample_rate)
plot_specgram(waveform, sample_rate)
play_audio(waveform, sample_rate)

plt.show()

# 日本語訳注
# 上記で使用しているplot_waveformなどの描画関数、
# play_audioという音声再生関数は
#
# セル「データとユーティリティ関数の準備
# このセルの中身を確認する必要はありません。
# 本セルを一度実行して、次のセルへと進んでください」
#
#を実行した際に定義されています

waveform, sample_rate = get_sample()
print_stats(waveform, sample_rate=sample_rate)

# エンコードを指定せずに保存する
# データに適合したエンコードが自動的に選択される
path = "save_example_default.wav"
torchaudio.save(path, waveform, sample_rate)
inspect_file(path)

# 16ビット符号付き整数リニアPCMとして保存する
# ファイルサイズは半分になりますが、精度が低下する
path = "save_example_PCM_S16.wav"
torchaudio.save(
    path, waveform, sample_rate,
    encoding="PCM_S", bits_per_sample=16)
inspect_file(path)




# ###
# waveform, sample_rate = get_sample()

# formats = [
#   "mp3",
#   "flac",
#   "vorbis",
#   "sph",
#   "amb",
#   "amr-nb",
#   "gsm",
# ]

# for format in formats:
#   path = f"save_example.{format}"
#   torchaudio.save(path, waveform, sample_rate, format=format)
#   inspect_file(path)

# データを読み込む
waveform1, sample_rate1 = get_sample(resample=16000)

# エフェクトを定義する
effects = [
  ["lowpass", "-1", "300"], # ローパスフィルター(単極)の適用
  ["speed", "0.8"],  # 速度を下げる
                     # サンプリングレートだけが変更されるので、 
                     # この後に元のサンプリングレートの値で
                     # `rate`エフェクトを適用する必要がある
  ["rate", f"{sample_rate1}"],
  ["reverb", "-w"],  # 反響のエフェクトを加えるとドラマチックな雰囲気になる
]

# エフェクトを適用する
waveform2, sample_rate2 = torchaudio.sox_effects.apply_effects_tensor(
    waveform1, sample_rate1, effects)

plot_waveform(waveform1, sample_rate1, title="Original", xlim=(-.1, 3.2))
plot_waveform(waveform2, sample_rate2, title="Effects Applied", xlim=(-.1, 3.2))
print_stats(waveform1, sample_rate=sample_rate1, src="Original")
print_stats(waveform2, sample_rate=sample_rate2, src="Effects Applied")

plot_specgram(waveform1, sample_rate1, title="Original", xlim=(0, 3.04))
play_audio(waveform1, sample_rate1)
plot_specgram(waveform2, sample_rate2, title="Effects Applied", xlim=(0, 3.04))
play_audio(waveform2, sample_rate2)