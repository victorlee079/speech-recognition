# Programming Assignment Part 5
import numpy as np
import librosa
import librosa.display
import math

sampling_rate = 48000
frame_size = 20
overlap_ratio = 0.5
wav_data_A = []
wav_data_B = []
np.set_printoptions(threshold=np.inf, linewidth=999)


def get_mfcc(data):
    result = []
    for i in data:
        n = int(frame_size / 1000 * sampling_rate)
        m = int(n / 2)
        result.append(librosa.feature.mfcc(y=i[0], sr=i[1], n_mfcc=13, hop_length=m, n_fft=n))
    return result


def get_distortion(source, target):
    dist_mat = np.zeros((source.shape[1], target.shape[1]))
    row, col = dist_mat.shape
    for x in range(row):
        for y in range(col):
            for i in range(1, 13):
                dist_mat[x][y] = dist_mat[x][y] + (source[i][x] - target[i][y]) ** 2
            dist_mat[x][y] = math.sqrt(dist_mat[x][y])
    for x in range(row):
        for y in range(col):
            args = []
            min_arg = 0.0
            if x > 0:
                args.append(dist_mat[x - 1][y])
            if y > 0:
                args.append(dist_mat[x][y - 1])
            if x > 0 and y > 0:
                args.append(dist_mat[x - 1][y - 1])
            if len(args) > 0:
                min_arg = min(args)
            dist_mat[x][y] = dist_mat[x][y] + min_arg
    return dist_mat


def get_accum_score(matrix):
    score = 0
    row, col = matrix.shape
    i = row - 1
    j = col - 1
    while i >= 0 and j >= 0:
        score += matrix[i][j]
        next_i, next_j = i, j
        next_val = 0
        if i > 0:
            next_val = matrix[i - 1][j]
            next_i, next_j = i - 1, j
        if j > 0 and matrix[i][j - 1] < next_val:
            next_val = matrix[i][j - 1]
            next_i, next_j = i, j - 1
        if i > 0 and j > 0 and matrix[i - 1][j - 1] < next_val:
            next_i, next_j = i - 1, j - 1
        if next_i == i and next_j == j:
            break
        i, j = next_i, next_j
    return score


# load data
for i in range(6):
    wav_data_A.append(librosa.load('./data/s' + str(i + 1) + 'A.wav', sr=48000))
    wav_data_B.append(librosa.load('./data/s' + str(i + 1) + 'B.wav', sr=48000))

mfcc_A = get_mfcc(wav_data_A)
mfcc_B = get_mfcc(wav_data_B)

comp_result = np.zeros((len(mfcc_A), len(mfcc_B)))

for a, a_mfcc in enumerate(mfcc_A):
    for b, b_mfcc in enumerate(mfcc_B):
        comp_result[a][b] = get_accum_score(get_distortion(a_mfcc, b_mfcc))

# nxn comparison result
rows, cols = comp_result.shape
x_headers = ["s" + str(i + 1) + "B" for i in range(6)]
y_headers = ["s" + str(i + 1) + "A" for i in range(6)]
print('\t{:6s}\t{:6s}\t{:6s}\t{:6s}\t{:6s}\t{:6s}'.format(*x_headers))
for i in range(rows):
    print(y_headers[i] + "\t" + "\t".join([str(int(round(y))) for y in comp_result[i]]))


# Optimal Path for S1A amd S1B
def mark_optimal_path(matrix):
    row, col = matrix.shape
    i = row - 1
    j = col - 1
    while i >= 0 and j >= 0:
        matrix[i][j] = 0 - matrix[i][j]
        next_i, next_j = i, j
        next_val = 0
        if i > 0:
            next_val = matrix[i - 1][j]
            next_i, next_j = i - 1, j
        if j > 0 and matrix[i][j - 1] < next_val:
            next_val = matrix[i][j - 1]
            next_i, next_j = i, j - 1
        if i > 0 and j > 0 and matrix[i - 1][j - 1] < next_val:
            next_i, next_j = i - 1, j - 1
        if next_i == i and next_j == j:
            break
        i, j = next_i, next_j


s1A_mfcc = mfcc_A[0]
s1B_mfcc = mfcc_B[0]
s1_dist = np.round(get_distortion(s1A_mfcc, s1B_mfcc))
mark_optimal_path(s1_dist)
s1_dist = np.flipud(s1_dist)
for row in s1_dist:
    for col in row:
        if col > 0:
            print(col, end="\t")
        else:
            print("*" + str(0 - col), end="\t")
    print()
