# Programming Assignment Part 4
from scipy.io import wavfile
from matplotlib import pyplot
import numpy as np
import math

fs, data = wavfile.read('./data/s1A.wav')

BITS = 16
ALPHA = 0.945
frame_size = 20
overlap_ratio = 0.5
# Samples in 1 frame
n = int(20 / 1000 * fs)
# Overlapping samples
m = int(n * overlap_ratio)


def get_frames(wav_data, frame_samples, overlap_size):
    start = 0
    end = 0
    result = []
    while end < len(wav_data):
        end = start + frame_samples
        if end > len(wav_data):
            end = len(wav_data)
        result.append(wav_data[start:start + frame_samples])
        start = start + overlap_size
    return result


def fourier_transform(input_samples):
    N = len(input_samples)
    x_reals = [0.0] * N
    x_imgs = [0.0] * N
    result = [0.0] * N
    for m in range(N):
        for k in range(N):
            s = input_samples[k]
            x_reals[m] = x_reals[m] + s * math.cos(2 * math.pi * k * m / N)
            x_imgs[m] = x_imgs[m] - s * math.sin(2 * math.pi * k * m / N)
        result[m] = math.sqrt(x_reals[m] ** 2 + x_imgs[m] ** 2)
    return result


def get_lpc(input_samples, order):
    r = [0.0] * (order + 1)
    for i in range(order + 1):
        for j in range(len(input_samples) - i):
            r[i] = r[i] + input_samples[j] * input_samples[j + i]
    correlation_matrix = [[]] * order
    for k in range(order):
        row = []
        for x in range(k, 0, -1):
            row.append(r[x])
        for y in range(0, order - k):
            row.append(r[y])
        correlation_matrix[k] = row
    part1 = np.linalg.inv(np.array(correlation_matrix))
    part2 = [[x] for x in r[1:]]
    result = np.matmul(part1, part2)
    lpc_parameters = [x[0] for x in result]
    return lpc_parameters


frames = get_frames(data, n, m)
num_of_frames = len(frames)
zcr = [0.0 for _ in range(num_of_frames)]
energy = [0.0 for _ in range(num_of_frames)]

for i in range(num_of_frames):
    frame = frames[i]
    zcr[i] = ((frame[:-1] * frame[1:]) < 0).sum()
    energy[i] = np.mean([x ** 2 for x in frame])

start_frame_index = -1
end_frame_index = -1
start_frame_count = 0
end_frame_count = 0

# By Observation
e_threshold = 0.1 * 10 ** 8
z_threshold = 400

for i in range(num_of_frames):
    z = zcr[i]
    e = energy[i]
    if start_frame_index < 0:
        if e > e_threshold and z > z_threshold:
            start_frame_count = start_frame_count + 1
        else:
            start_frame_count = 0

    if start_frame_index < 0 and start_frame_count >= 3:
        start_frame_index = i - 2

    if start_frame_index > -1 and end_frame_index < 0:
        if e < e_threshold and z < z_threshold:
            end_frame_count = end_frame_count + 1
        else:
            end_frame_count = 0

    if end_frame_index < 0 and end_frame_count >= 5:
        end_frame_index = i - 4
# For start index is after end index
if end_frame_index < start_frame_index:
    end_frame_index = num_of_frames - 1

start_index = start_frame_index * m
end_index = end_frame_index * m

t1 = start_index * 1 / fs
t2 = end_index * 1 / fs

print("T1: " + str(t1))
print("T2: " + str(t2))

speech_data_frames = frames[start_frame_index:end_frame_index]
seg1 = speech_data_frames[4]
seg1_start_index = (start_frame_index + 4) * m
seg1_end_index = seg1_start_index + n

formatted_seg1 = [x + 2 ** BITS / 2 for x in seg1]
ft_seg1 = fourier_transform(formatted_seg1)

fft_temp = sorted(zip(formatted_seg1, ft_seg1))
sort_x = [i[0] for i in fft_temp]
sort_y = [i[1] for i in fft_temp]

pem_seg1 = [0.0 for i in range(n - 1)]
for i in range(n - 1):
    pem_seg1[i] = formatted_seg1[i + 1] - ALPHA * formatted_seg1[i]

lpc_params = get_lpc(pem_seg1, 10)
print("LPC: " + str(lpc_params))

pyplot.figure(figsize=(8, 6))
pyplot.subplot(311)
pyplot.plot(data)
pyplot.axvline(x=start_index, color="r", label="T1: " + str(t1))
pyplot.axvline(x=end_index, color="b", label="T2: " + str(t2))
pyplot.axvline(x=seg1_start_index, color="black", label="Seg1")
pyplot.axvline(x=seg1_end_index, color="black")
pyplot.legend()
pyplot.subplot(312)
pyplot.plot(zcr)
pyplot.ylabel("Zero Crossing Rate")
pyplot.subplot(313)
pyplot.plot(energy)
pyplot.ylabel("Energy")
pyplot.tight_layout()

pyplot.figure(figsize=(8, 6))
pyplot.subplot(311)
pyplot.title("Seg1")
pyplot.plot(formatted_seg1)
pyplot.subplot(312)
pyplot.title("Pem_Seg1")
pyplot.plot(pem_seg1)
pyplot.subplot(313)
pyplot.title("Fourier Transform of Seg1")
pyplot.plot(sort_x, sort_y, scalex=True)
pyplot.xlabel("frequency")
pyplot.ylabel("energy")
pyplot.tight_layout()

pyplot.show()
