import wave
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from scipy.fft import fft
from scipy.signal import welch

# Function to convert MP3 to WAV using ffmpeg
def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    subprocess.run(['ffmpeg', '-i', mp3_file_path, wav_file_path])

# Function to read wave file and return signal and frame rate
def read_wave_file(file_path):
    with wave.open(file_path, 'r') as wav_file:
        frames = wav_file.readframes(-1)
        signal = np.frombuffer(frames, dtype=np.int16)
        frame_rate = wav_file.getframerate()
    return signal, frame_rate

# Function to compute the power spectral density using Welch's method
def power_spectral_density(signal, frame_rate):
    freqs, psd = welch(signal, frame_rate)
    return freqs, psd

# File paths for the MP3 and converted WAV files
file_original = r"original.mp3"
file_painted = r'painted.mp3'
wav_file_original = 'original.wav'
wav_file_painted = 'painted.wav'

# Convert the MP3 files to WAV
convert_mp3_to_wav(file_original, wav_file_original)
convert_mp3_to_wav(file_painted, wav_file_painted)

# Reading the converted WAV files
signal_original, frame_rate_original = read_wave_file(wav_file_original)
signal_painted, frame_rate_painted = read_wave_file(wav_file_painted)

# Time axes for plotting waveforms
time_original = np.linspace(0, len(signal_original) / frame_rate_original, num=len(signal_original))
time_painted = np.linspace(0, len(signal_painted) / frame_rate_painted, num=len(signal_painted))

# Plotting the waveforms
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time_original, signal_original)
plt.title("Waveform of Original Sound")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.subplot(2, 1, 2)
plt.plot(time_painted, signal_painted)
plt.title("Waveform of Repainted Sound")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.tight_layout()

# Computing FFT and power spectral density for both signals
fft_original = fft(signal_original)
fft_painted = fft(signal_painted)
freqs_original, psd_original = power_spectral_density(signal_original, frame_rate_original)
freqs_painted, psd_painted = power_spectral_density(signal_painted, frame_rate_painted)

# Plotting FFT and power spectral density
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(np.abs(fft_original)[:len(fft_original)//2])
plt.title("FFT of Original Sound")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.subplot(2, 2, 2)
plt.plot(np.abs(fft_painted)[:len(fft_painted)//2])
plt.title("FFT of Repainted Sound")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.subplot(2, 2, 3)
plt.semilogy(freqs_original, psd_original)
plt.title("Power Spectral Density of Original Sound")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [V**2/Hz]")
plt.subplot(2, 2, 4)
plt.semilogy(freqs_painted, psd_painted)
plt.title("Power Spectral Density of Repainted Sound")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [V**2/Hz]")
plt.tight_layout()
plt.show()
