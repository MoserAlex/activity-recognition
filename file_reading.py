import csv
import numpy as np
import os
from enums.activities import Activities, Index
from enums.features import Features
from enums.sensorPosition import SensorPosition

g_nr_of_training_subjects = 5
g_test_subject = 0
g_selected_activities = list()
g_selected_features = list()
g_sensor_position = SensorPosition.CLOTH_POCKET.value

def setup(nr_of_training_subjects=5, test_subject=0, selected_activities=list(), selected_features=list(), sensor_position=SensorPosition.CLOTH_POCKET):
  global g_nr_of_training_subjects
  global g_selected_activities
  global g_selected_features
  global g_sensor_position
  global g_test_subject

  g_nr_of_training_subjects = nr_of_training_subjects
  g_test_subject = test_subject
  g_selected_activities = selected_activities
  g_selected_features = selected_features
  g_sensor_position = sensor_position.value

# ------------------------------------------------------------ #
#                                                              #
#                      Reading Data Files                      #
#                                                              #
# ------------------------------------------------------------ #

def read_data_files(root: str, look_for_test_subject=False, data=None, categories=None, log=''):
  if data == None:
    data = list()
  if categories == None:
    categories = list()

  names = __get_file_names(root)
  root = root + '/'
  count = 0

  for i, name in enumerate(names):
    path = root + name

    # recursive call if it's a directory
    if os.path.isdir(path):
      if __skip_subject(i, count, look_for_test_subject):
        continue

      count += 1
      log = '  Directory {} '.format(count)

      if look_for_test_subject:
        log = '  Directory {} '.format(i + 1)

      # recursive call if all checks pass
      read_data_files(path, look_for_test_subject, data, categories, log)

    # start reading if it's a file
    if os.path.isfile(path):
      fileData, activity = read_data_file(path, log, i)

      if len(fileData) > 0:
        categories.append(activity)
        data.append(fileData)

  return data, categories

def read_data_file(path: str, log='', index=-1):
  data = list()
  route = path.split('/')
  filename = route[len(route)-1]
  split_name = filename.split('_')

  activity = split_name[0]
  if len(g_selected_activities) > 0 and Activities(activity) not in g_selected_activities:
    return data, activity

  position = split_name[1]
  if position != g_sensor_position:
    return data, activity

  if index != -1:
    print(log + ' - reading file {}'.format(index), end='\r')

  data = __get_file_data(path)
  return data, activity

# reads the data of current file
def __get_file_data(path: str):
  samples = list()
  with open(path) as csvfile:
    readNP = np.loadtxt(csvfile, delimiter=',')
    for row in readNP:
      if len(row) == 3:
        newSample = [row[0], row[1], row[2]]
        samples.append(newSample)

  return samples

# ----------------------------------------------------------- #
#                                                             #
#                    Reading Feature Files                    #
#                                                             #
# ----------------------------------------------------------- #

def read_feature_files(root: str, look_for_test_subject=False):
  root = root + '/' + g_sensor_position
  data = list()
  categories = list()
  file_names = __get_file_names(root)

  activity_indizes = list()
  for activity in g_selected_activities:
    activity_indizes.append(Index(activity))

  count = 0

  for i, file_name in enumerate(file_names):
    if __skip_subject(i, count, look_for_test_subject):
      continue

    count += 1

    path = root + '/' + file_name

    with open(path, 'r') as csvfile:
      reader = np.loadtxt(csvfile, delimiter=',')

      for row in reader:
        # check if the activity is one we want
        if len(activity_indizes) == 0 or row[0] in activity_indizes:
          categories.append(row[0])
          sample = __discard_unwanted_features(row[1 : len(row)])
          data.append(sample)

  return data, categories

def __discard_unwanted_features(feature_sample: list):
  if len(g_selected_features) == 0:
    return feature_sample

  sample = list()
  h_sample = list()
  v_sample = list()

  count = len(feature_sample) // 3
  for i in range(count):
    if Features(i) in g_selected_features:
      if Features.absolute in g_selected_features or Features.horizontal in g_selected_features or Features.vertical in g_selected_features:
        if Features.absolute in g_selected_features:
          sample.append(feature_sample[i])
        if Features.horizontal in g_selected_features:
          h_sample.append(feature_sample[i + count])
        if Features.vertical in g_selected_features:
          v_sample.append(feature_sample[i + 2 * count])
      # if it's not specified which part(absolute, horizontal, vertical) of the features should be used, use all of them
      else:
        sample.append(feature_sample[i])
        h_sample.append(feature_sample[i + count])
        v_sample.append(feature_sample[i + 2 * count])

  sample.extend(h_sample)
  sample.extend(v_sample)
  return sample

# ------------------------------------------------------------- #
#                                                               #
#                            Utility                            #
#                                                               #
# ------------------------------------------------------------- #

# returns all file names in current directory
def __get_file_names(path: str):
  names = list()
  ignore = list(['readme.txt', 'INFORMATION.txt'])
  dirlist = os.listdir(path)
  for filename in dirlist:
    if filename not in ignore:
      names.append(filename)

  return names

def __skip_subject(index: int, count: int, search: bool):
  # look for the test subject and skip all others
  if search and index != g_test_subject:
    return True

  # skip the test subject when reading all other files
  if not search and index == g_test_subject:
    return True

  # don't read too many files
  if count >= g_nr_of_training_subjects:
    return True
  
  return False