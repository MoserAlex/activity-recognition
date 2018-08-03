import feature_extraction as extractor
import file_reading as reader
import machine_learning as machine
import report
from enums.activities import Activities
from enums.sensorPosition import SensorPosition
from enums.features import Features
import time

def start(nr_of_training_subjects: int, test_subject: int, activities, features, sensor_position: SensorPosition, cluster_size: int):
  report.start()
  report.setup(nr_of_training_subjects, test_subject, activities, features, sensor_position, cluster_size)

  reader.setup(
    nr_of_training_subjects=nr_of_training_subjects,
    test_subject=test_subject,
    selected_activities=activities,
    selected_features=features,
    sensor_position=sensor_position,
    cluster_size=cluster_size
  )

  report.add('\nTraining:')
  training_features, training_targets = get_features_from_reader()
  machine.Train(training_features, training_targets, cross_validation=True)

  report.add()

  report.add('\nClassification:')
  test_features, test_targets = get_features_from_reader(is_for_training=True)
  machine.Classification(test_features, test_targets)

  report.stop()

def get_features_from_reader(use_feature_files=True, is_for_training=False):
  features = list()
  targets = list()

  if use_feature_files:
    report.add('Reading feature files', indentation=1)
    start = time.time()
    features, targets = reader.read_feature_files('FeatureSets', is_for_training)
    report.add(' - finished in {0:.2f} seconds\n'.format(time.time()-start), indentation=1)

  else:
    report.add('Reading data files', indentation=1)
    start = time.time()
    files, labels = reader.read_data_files('DataSets', is_for_training)
    report.add(' - finished in {0:.2f} seconds\n'.format(time.time()-start), indentation=1)

    report.add('Calculating features', indentation=1)
    start = time.time()
    features, targets = extractor.GetFeatures(files, labels, reader.g_selected_features)
    report.add(' - finished in {0:.2f} seconds\n'.format(time.time()-start), indentation=1)

  return features, targets

activities = [
  Activities.BICYCLE,
  Activities.CLIMB_DOWN,
  Activities.CLIMB_UP,
  Activities.JUMP,
  Activities.RELAX,
  Activities.RUN,
  Activities.WALK,
  # Activities.WALK_BACK,
  # Activities.WALK_QUICK,
  # Activities.WALK_STEP,
  # Activities.ELEVATOR,
  # Activities.TELEPHONE,
]
features = [
  Features.MEAN_AMPLITUDE,
  Features.MAX_AMPLITUDE,
  Features.MIN_AMPLITUDE,
  Features.VARIANCE,
  Features.STANDARD_DEVIATION,
  Features.ENERGY_AMPLITUDE,
  Features.STFT_AMPLITUDE,
  Features.HJORTH_MOBILITY,
  Features.HJORTH_COMPLEXITY,

  # # if these are specified, only the corresponding value of each feature will be used.
  Features.absolute,
  Features.horizontal,
  Features.vertical,
]
positions = [
  # SensorPosition.CLOTH_POCKET,
  SensorPosition.TROUSERS_POCKET,
  # SensorPosition.WAIST
]
clusters = [
  64,
  # 256,
  # 512,
  # 1024,
]
subjects = [
  # 1,
  # 5,
  # 10,
  # 20,
  44,
]
test_subject = 5

# Here the execution of the main code starts
for p in positions:
  for c in clusters:
    for s in subjects:
      start(s, test_subject, activities, features, p, c)
