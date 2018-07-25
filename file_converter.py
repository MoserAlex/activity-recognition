import csv
import os
import time
import feature_extraction as extractor
import file_reading as reader
from enums.sensorPosition import SensorPosition

def convert_data_to_feature_file(data_root: str, write_directory: str, skip_existing_files=False):  
  subjects = __getFileNames(data_root)

  feature_root = 'FeatureSets'
  write_directory = feature_root + '/' + write_directory + '/'

  __create_folder(feature_root)
  __create_folder(write_directory)

  existing_files = os.listdir(write_directory)

  for i, subject in enumerate(subjects):
    file_name = subject + '.csv'
    if file_name in existing_files:
      continue

    print(' -> converting {}. subject: {}'.format(i + 1, subject))

    sub_path = data_root + '/' + subject
    file_names = __getFileNames(sub_path)

    data = list()
    labels = list()
    for name in file_names:
      full_path = sub_path + '/' + name
      # print(full_path)
      training_files, training_labels = reader.read_data_file(full_path)
      if len(training_files) > 0:
        data.append(training_files)
        labels.append(training_labels)

    features, targets = extractor.GetFeatures(data, labels)
    __write_feature_file(write_directory + file_name, features, targets)

def __create_folder(root: str):
  if not os.path.exists(root):
    os.mkdir(root)

# returns all file names in current directory
def __getFileNames(path: str):
  names = list()
  ignore = list(['readme.txt', 'INFORMATION.txt'])
  dirlist = os.listdir(path)
  for filename in dirlist:
    if filename not in ignore:
      names.append(filename)

  return names

def __write_feature_file(path: str, data: list, targets: list):
  with open(path, 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    for i, features in enumerate(data):
      row = list(features)
      row.insert(0, targets[i])
      writer.writerow(row)



def start_conversion(sensor_position):
  print('Converting data files - {}'.format(sensor_position.value))
  reader.setup(
    sensor_position=sensor_position
  )
  start = time.time()
  convert_data_to_feature_file('DataSets', sensor_position.value, skip_existing_files=True)
  print(' - finished in {0:.2f} seconds\n'.format(time.time()-start))

start_conversion(SensorPosition.CLOTH_POCKET)
start_conversion(SensorPosition.TROUSERS_POCKET)
start_conversion(SensorPosition.WAIST)