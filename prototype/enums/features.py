from enum import Enum

class Features(Enum):
  # The number corresponds to the index of the feature in the files after the first column (target) is extracted
  # Don't change the order unless the feature sets are being recalculated
  MEAN_AMPLITUDE = 0
  MAX_AMPLITUDE = 1
  MIN_AMPLITUDE = 2
  VARIANCE = 3
  STANDARD_DEVIATION = 4
  ENERGY_AMPLITUDE = 5
  STFT_AMPLITUDE = 6
  HJORTH_MOBILITY = 7
  HJORTH_COMPLEXITY = 8

  absolute = -1
  horizontal = -2
  vertical = -3

FeatureDictionary = {
  Features.MEAN_AMPLITUDE: 'Mean Amplitude',
  Features.MAX_AMPLITUDE: 'Max Amplitude',
  Features.MIN_AMPLITUDE: 'Min Amplitude',
  Features.VARIANCE: 'Variance',
  Features.STANDARD_DEVIATION: 'Standard Deviation',
  Features.ENERGY_AMPLITUDE: 'Energy Amplitude',
  Features.STFT_AMPLITUDE: 'Short Time Fourier Transformation',
  Features.HJORTH_MOBILITY: 'Hjorth Mobility',
  Features.HJORTH_COMPLEXITY: 'Hjorth Complexity',
  
  Features.absolute: 'absolute',
  Features.horizontal: 'horizontal',
  Features.vertical: 'vertical',
}