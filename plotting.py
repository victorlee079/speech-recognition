# Programming Assignment Part 3
from scipy.io import wavfile
from matplotlib import pyplot
import numpy as np

fs, data = wavfile.read('./data/s1A.wav')

formatted_data = [x + 32768 for x in data]
times = np.arange(0, 1 / fs * len(data) * 1000, 1 / fs * 1000)

pyplot.plot(times, formatted_data)
pyplot.xlabel("Time in ms")
pyplot.ylabel("Pressure or Voltage")
pyplot.tight_layout()
pyplot.show()
