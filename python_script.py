# importing all the required libraries
import numpy as np
import wave                                 # To open audio files
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy.io import wavfile                # to export the result files

track_1 = wave.open('./Samples/sample_1.wav', 'r')
track_2 = wave.open('./Samples/sample_2.wav', 'r')

track_1.getparams()

track_2.getparams()

# Both tracks have:
# - Framerate: 48000
# - Total Frames: 203776
# - Channels: 1

# length of samples:
framerate = 48000
total_frames = 203776

print(f"Track time in seconds: {total_frames/framerate}")

_, signal_1 = wavfile.read('./Samples/sample_1.wav')
_, signal_2 = wavfile.read('./Samples/sample_2.wav')

print("shape of signal 1: ", signal_1.shape)
print("shape of signal 2: ", signal_2.shape)

print()
print("First 10 values of signal 1:")
print(signal_1[:10])
print()
print("First 10 values of signal 2:")
print(signal_2[:10])

# Plotting each signal
fs = track_1.getframerate()
timing = np.linspace(0, len(signal_1)/fs, len(signal_1))

plt.figure(figsize=(12, 2))
plt.title("Signal 1")
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.plot(timing, signal_1, c='#3ABFE7')
plt.ylim(-30000, 30000)
plt.show()

plt.figure(figsize=(12, 2))
plt.title("Signal 2")
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.plot(timing, signal_2, c='#df8efd')
plt.ylim(-30000, 30000)
plt.show()

# Zipping signals together
X = list(zip(signal_1, signal_2))

X[:10]

# Model
ica = FastICA(n_components=2)

results = ica.fit_transform(X)

result_signal_1 = results[:, 0]
result_signal_2 = results[:, 1]

# Plotting the result signals
plt.figure(figsize=(12, 2))
plt.title("Independent Component # 1")
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.plot(timing, result_signal_1, c='#3ABFE7')
plt.show()

plt.figure(figsize=(12, 2))
plt.title("Independent Component # 2")
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.plot(timing, result_signal_2, c='#df8efd')
plt.show()

# Saving the results:
# Convert to int, map the appropriate range, and increase the volume a little bit
scaling_factor = 32767 / 5.0
result_signal_1_int = np.int16(result_signal_1 * scaling_factor)
result_signal_2_int = np.int16(result_signal_2 * scaling_factor)

# Write wave files
wavfile.write("./Results/Independent_Component_1.wav", fs, result_signal_1_int)
wavfile.write("./Results/Independent_Component_2.wav", fs, result_signal_2_int)
