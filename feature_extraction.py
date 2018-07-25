from sklearn import preprocessing
import numpy as np
import scipy
from enums.activities import Activities, Index
from enums.features import Features
import sys

g_smoothing_points = 5
g_sample_size = 64

def GetFeatures(data: list, label: list, selectedFeatures=list()):
  features = list()
  targets = list()

  for i, activity in enumerate(data):
    print('  calc activity features {}/{}'.format(i, len(data)), end='\r')
    magnitudes, horizontal, vertical = __get_magnitudes(activity)

    __smooth_data(magnitudes)
    __smooth_data(horizontal)
    __smooth_data(vertical)

    feats = __calculate_features(magnitudes, selectedFeatures, g_sample_size)
    h_feats = __calculate_features(horizontal, selectedFeatures, g_sample_size)
    v_feats = __calculate_features(vertical, selectedFeatures, g_sample_size)

    # combine corresponding features
    feats = np.append(feats, h_feats, axis=0)
    feats = np.append(feats, v_feats, axis=0)
    feats = np.transpose(feats)

    feats = __remove_invalid_features(feats)
    tar = np.full(len(feats), Index(Activities(label[i])))

    features.extend(feats)
    targets.extend(tar)

  return features, targets

def __get_magnitudes(data: list):
  magnitudes = list()
  horizontal = list()
  vertical = list()

  for x, y, z in data:
    magnitudes.append(np.sqrt(x*x + y*y + z*z))
    horizontal.append(np.sqrt(x*x + y*y))
    vertical.append(z)

  return magnitudes, horizontal, vertical

def __smooth_data(data: list, points = g_smoothing_points):
  reach = points // 2
  for i in range(len(data) - reach * 2):
    sum = 0
    for j in range(reach * -1, reach + 1):
      sum += data[i + j]
    data[i] = sum / points

def __calculate_features(data: list, selectedFeatures: list, fftsize=256, overlap=2):
  hop = fftsize // overlap
  features = list()

  if (len(selectedFeatures) == 0 or Features.MEAN_AMPLITUDE in selectedFeatures):
    features.append(__mean_amplitude(data, fftsize, hop))

  if (len(selectedFeatures) == 0 or Features.MAX_AMPLITUDE in selectedFeatures):
    features.append(__max_amplitude(data, fftsize, hop))

  if (len(selectedFeatures) == 0 or Features.MIN_AMPLITUDE in selectedFeatures):
    features.append(__min_amplitude(data, fftsize, hop))

  if (len(selectedFeatures) == 0 or Features.VARIANCE in selectedFeatures):
    features.append(__variance(data, fftsize, hop))

  if (len(selectedFeatures) == 0 or Features.STANDARD_DEVIATION in selectedFeatures):
    features.append(__standard_deviation(data, fftsize, hop))

  if (len(selectedFeatures) == 0 or Features.ENERGY_AMPLITUDE in selectedFeatures):
    features.append(__energy_amplitude(data, fftsize, hop))

  if (len(selectedFeatures) == 0 or Features.STFT_AMPLITUDE in selectedFeatures):
    features.append(__stft(data, fftsize, hop))

  if (len(selectedFeatures) == 0 or Features.HJORTH_MOBILITY in selectedFeatures or Features.HJORTH_COMPLEXITY in selectedFeatures):
    mobility, complexity = __hjorth_parameter(data, fftsize, hop)
    
    if (len(selectedFeatures) == 0 or Features.HJORTH_MOBILITY):
      features.append(mobility)
    if (len(selectedFeatures) == 0 or Features.HJORTH_COMPLEXITY):
      features.append(complexity)

  return features

def __remove_invalid_features(features: list):
  corrected = list()
  for i in range(len(features)):
    # skip samples, which contain and infinity-value and are not useable
    if float('inf') not in features[i]:
      corrected.append(features[i])

  return corrected

# ------------------------------------------------------------ #
#                                                              #
#                           Features                           #
#                                                              #
# ------------------------------------------------------------ #

def __mean_amplitude(X: list, fftsize: int, hop: int):
  feats = list()
  for i in range(0, len(X) - fftsize, hop):
    feats.append(np.mean(X[i : i + fftsize]))
  return feats

def __max_amplitude(X: list, fftsize: int, hop: int):
  feats = list()
  for i in range(0, len(X) - fftsize, hop):
    feats.append(np.max(X[i : i + fftsize]))
  return feats

def __min_amplitude(X: list, fftsize: int, hop: int):
  feats = list()
  for i in range(0, len(X) - fftsize, hop):
    feats.append(np.min(X[i : i + fftsize]))
  return feats

def __variance(X: list, fftsize: int, hop: int):
  feats = list()
  for i in range(0, len(X) - fftsize, hop):
    feats.append(np.var(X[i : i + fftsize]))
  return feats

def __standard_deviation(X: list, fftsize: int, hop: int):
  feats = list()
  for i in range(0, len(X) - fftsize, hop):
    feats.append(np.std(X[i : i + fftsize]))
  return feats

def __energy_amplitude(X: list, fftsize: int, hop: int):
  feats = list()
  for i in range(0, len(X) - fftsize, hop):
    feats.append(np.sum(np.power(X[i : i + fftsize], 2)))
  return feats

def __stft(X: list, fftsize: int, hop: int):
  w = scipy.hanning(fftsize + 1)[: -1]
  energy_amplitude = list()

  stft_signal = np.array([np.fft.rfft(w * X[i : i + fftsize]) for i in range(0, len(X) - fftsize, hop)])

  for i, val in enumerate(stft_signal):
    energy_amplitude.append(np.sum(np.power(abs(val),2)))

  return energy_amplitude

def __hjorth_parameter(X: list, n: int, hop: int):
  mobility = list()
  complexity = list()

  D = list()
  # Calculate first order diff
  for i in range(1,len(X)):
    D.append(X[i]-X[i-1])

  D.insert(0, np.mean(D[0 : n])) # pad the first difference
  D = np.array(D)

  for i in range(0, len(X) - n, hop):
    tmp_X = X[i : i + n]
    tmp_D = D[i : i + n]

    M2 = float(sum(tmp_D ** 2)) / n
    TP = sum(np.array(tmp_X) ** 2)
    M4 = 0
    for j in range(1, len(tmp_D)):
      M4 += (tmp_D[j] - tmp_D[j - 1]) ** 2
    M4 = M4 / n

    if TP == 0:
      # this value can be found later and the whole sample will be removed from the set
      mobility.append(float('inf'))
    else:
      mobility.append(np.sqrt(M2 / TP))

    if M2 == 0:
      # this value can be found later and the whole sample will be removed from the set
      complexity.append(float('inf'))
    else:
      complexity.append(np.sqrt(float(M4) * TP / M2 / M2))

  return mobility, complexity