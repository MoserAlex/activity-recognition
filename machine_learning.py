from sklearn import model_selection, preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import time
import report

g_decision_tree = None
g_knn = None
g_linear_svm = None
g_rbf_svm = list()

g_scaler = None

def Train(data: list, targets: list, cross_validation=False):
  report.add('Start Training', indentation=1)
  start = time.time()

  X_train, X_test, y_train, y_test = model_selection.train_test_split(data, targets, random_state=0)
  # Configure Scaler
  global g_scaler
  g_scaler = preprocessing.StandardScaler().fit(X_train)
  X_train = g_scaler.transform(X_train)

  __train_decision_tree(X_train, y_train)
  __train_knn(X_train, y_train)
  __train_linear_svm(X_train, y_train)
  __train_rbf_svm(X_train, y_train)

  report.add('End Training', indentation=1)
  report.add(' - finished in {0:.2f} seconds'.format(time.time() - start), indentation=1)

  if cross_validation:
    report.add()
    Classification(X_test, y_test, 'Cross Validation')

def Classification(data: list, targets: list, log='Classification'):
  report.add('Start {}'.format(log), indentation=1)
  report.add('Accuracy:', indentation=2)

  X_test = data
  if g_scaler is not None:
    X_test = g_scaler.transform(data)

  __test_decision_tree(X_test, targets)
  __test_knn(X_test, targets)
  __test_linear_svm(X_test, targets)
  __test_rbf_svm(X_test, targets)

  report.add('End {}'.format(log), indentation=1)

# ------------------------------------------------------------ #
#                                                              #
#                       Train Classifier                       #
#                                                              #
# ------------------------------------------------------------ #

def __train_decision_tree(X_train: list, y_train: list):
  report.add('Decision Tree', indentation=2, end = '', flush=True)
  start = time.time()

  dt = DecisionTreeClassifier(max_depth=2)
  dtree_model = dt.fit(X_train, y_train)

  global g_decision_tree
  g_decision_tree = dtree_model

  report.add('          time = {0:.2f} seconds'.format(time.time()-start))

def __train_knn(X_train: list, y_train: list):
  report.add('K-Nearest Neighbor', indentation=2, end = '', flush=True)
  start = time.time()

  knn = KNeighborsClassifier()
  knn_model = knn.fit(X_train, y_train)

  global g_knn
  g_knn = knn_model

  report.add('     time = {0:.2f} seconds'.format(time.time()-start))

def __train_linear_svm(X_train: list, y_train: list):
  report.add('SVM with Linear Kernel', indentation=2, end = '', flush=True)
  start = time.time()

  svm = SVC(kernel='linear', C=1.0)
  svm_model_linear = svm.fit(X_train, y_train)

  global g_linear_svm
  g_linear_svm = svm_model_linear

  report.add(' time = {0:.2f} seconds'.format(time.time()-start))

def __train_rbf_svm(X_train: list, y_train: list):
  report.add('SVM with RBF Kernel', indentation=2)

  C_2d_range = [1, 1e2]
  gamma_2d_range = [1e-2, 1e-1, 1]
  classifiers = list()
  for C in C_2d_range:
    for gamma in gamma_2d_range:
      print('      {}. rbf:  C = {}  gamma = {}'.format(len(classifiers) + 1, C, gamma), end = '', flush=True)
      start = time.time()

      svm = SVC(gamma=gamma, C=C)
      svm_model_rbf = svm.fit(X_train, y_train)
      classifiers.append(svm_model_rbf)

      print('  time = {0:.2f} seconds'.format(time.time()-start))

      report.add('{}. rbf:             '.format(len(classifiers)), indentation=3, end = '', in_console=False)
      report.add(' time = {0:.2f} seconds'.format(time.time()-start), end = '', in_console=False)
      report.add(' C = {}  gamma = {}'.format(C, gamma), in_console=False)

  global g_rbf_svm
  g_rbf_svm = classifiers

# ----------------------------------------------------------- #
#                                                             #
#                       Test Classifier                       #
#                                                             #
# ----------------------------------------------------------- #

def __test_decision_tree(X_test, targets):
  if g_decision_tree is not None:
    accuracy = g_decision_tree.score(X_test, targets) * 100

    report.add('Decision Tree:      {0:.2f}%'.format(accuracy), indentation=3)

    # prediction = g_decision_tree.predict(X_test)
    # con_matrix = confusion_matrix(targets, prediction)

def __test_knn(X_test, targets):
  if g_knn is not None:
    accuracy = g_knn.score(X_test, targets) * 100

    report.add('K-Nearest Neighbor: {0:.2f}%'.format(accuracy), indentation=3)

    # prediction = g_knn.predict(X_test)
    # con_matrix = confusion_matrix(targets, prediction)

def __test_linear_svm(X_test, targets):
  if g_linear_svm is not None:
    accuracy = g_linear_svm.score(X_test, targets) * 100

    report.add('Linear SVM:         {0:.2f}%'.format(accuracy), indentation=3)

    # prediction = g_linear_svm.predict(X_test)
    # con_matrix = confusion_matrix(targets, prediction)

def __test_rbf_svm(X_test, targets):
  if len(g_rbf_svm) is not 0:
    report.add('RBF SVM:', indentation=3)

    for i, classifier in enumerate(g_rbf_svm):
      accuracy = classifier.score(X_test, targets) * 100

      msg = '{}. rbf:           '.format(i + 1)
      msg = msg + '{0:.2f}% '.format(accuracy)
      msg = msg + ' C = {}  gamma = {}'.format(classifier.C, classifier.gamma)

      report.add(msg, indentation=4)

      # prediction = classifier.predict(X_test)
      # con_matrix = confusion_matrix(targets, prediction)
