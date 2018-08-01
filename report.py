import datetime
import os
from enums.activities import Activities
from enums.features import Features, FeatureDictionary
from enums.sensorPosition import SensorPosition

g_report_root_directory = 'Reports'

g_report_title = 'report'
g_report_text = ''
g_report_indentation = '  '

def start(custom_title=''):
  __create_report_directory()
  __set_file_name(custom_title)
  __set_first_line()

def stop():
  path = g_report_root_directory + '/' + g_report_title

  with open(path, 'w') as report:
    report.write(g_report_text)

def setup(training_subjects: int, test_subject: int, activities: Activities, features: Features, sensor_position: SensorPosition, cluster_size: int):
  __setup_subjects(training_subjects, test_subject)
  __setup_sensor(sensor_position)
  __setup_activities(activities)
  __setup_features(features, cluster_size)

def add(line='', indentation=0, end='\n', flush=False, in_console=True):
  global g_report_text
  indent = g_report_indentation * indentation

  if in_console:
    print(indent + line, end=end, flush=flush)

  g_report_text += indent + line + end

# ------------------------------------------------------------- #
#                                                               #
#                            Utility                            #
#                                                               #
# ------------------------------------------------------------- #

def __create_report_directory():
  if not os.path.exists(g_report_root_directory):
    os.mkdir(g_report_root_directory)

def __set_file_name(custom_title):
  global g_report_title

  now = datetime.datetime.now()
  date = str(now.date()).replace('-', '.')
  time = str(now.time()).split('.')[0].replace(':', '-')

  g_report_title = date + '__'
  if custom_title != '':
    g_report_title += custom_title + '__'
  else:
    g_report_title += 'Report' + '__'
  g_report_title += time + '.txt'

def __set_first_line():
  global g_report_text

  now = datetime.datetime.now()
  date = str(now.date()).replace('-', '.')
  time = str(now.time()).split('.')[0]

  g_report_text = 'Report - ' + date + ' ' + time + '\n'

  g_report_text += '\n'

def __setup_subjects(training_subjects: int, test_subject: int):
  global g_report_text

  g_report_text += 'The machine is trained with the data of {} training subject'.format(min(43, training_subjects))
  if training_subjects > 1:
    g_report_text += 's'
  g_report_text += '.\n'
  g_report_text += 'The activities of test subject {} are being classified.\n'.format(test_subject + 1)

  g_report_text += '\n'

def __setup_sensor(sensor_position: SensorPosition):
  global g_report_text

  g_report_text += 'The sensor is positioned at the {}.\n'.format(sensor_position.value.title())

  g_report_text += '\n'

def __setup_activities(activities: Activities):
  global g_report_text

  g_report_text += 'The following Activities are being classified:\n'
  for act in activities:
    g_report_text += g_report_indentation + act.value.title() + '\n'

  g_report_text += '\n'

def __setup_features(features: Features, cluster_size: int):
  global g_report_text

  # List if the absolute, vertical or horizontal component of the features are used
  g_report_text += 'The features were calculated from clusters of size {} of raw acceleration data.\n'.format(str(cluster_size))
  g_report_text += 'Of the following Features the '
  components = list()
  for feat in features:
    if feat is Features.absolute or feat is Features.horizontal or feat is Features.vertical:
      components.append(FeatureDictionary[feat])

  if len(components) == 0 or len(components) == 3:
    g_report_text += '{}, {} and {} components are '.format(FeatureDictionary[Features.absolute], FeatureDictionary[Features.horizontal], FeatureDictionary[Features.vertical])
  elif len(components) == 2:
    g_report_text += '{} and {} components are '.format(components[0], components[1])
  elif len(components) == 1:
    g_report_text += '{} component is '.format(components[0])

  g_report_text += 'used for comparison:\n'

  # List all Features
  for feat in features:
    if feat is not Features.absolute and feat is not Features.horizontal and feat is not Features.vertical:
      g_report_text += g_report_indentation + FeatureDictionary[feat].title() + '\n'

  g_report_text += '\n'
