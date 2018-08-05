from enum import Enum

class Activities(Enum):
  # These string are being used to know which data file belongs to which activity
  # Don't change them, unless you rename all data files
  BICYCLE = 'bicycling'
  CLIMB_DOWN = 'climbing downstairs'
  CLIMB_UP = 'climbing upstairs'
  JUMP = 'jump 45 seconds'
  RELAX = 'releax'
  RUN = 'run 100m'
  WALK = 'walking 50m'
  WALK_BACK = 'walking backward'
  WALK_QUICK = 'walking quickly 50m'
  WALK_STEP = 'step walking 45s'
  ELEVATOR = 'riding elavator upward'
  TELEPHONE = 'take phone'

_indexDictionary = {
  Activities.BICYCLE: 0,
  Activities.CLIMB_DOWN: 1,
  Activities.CLIMB_UP: 2,
  Activities.JUMP: 3,
  Activities.RELAX: 4,
  Activities.RUN: 5,
  Activities.WALK: 6,
  Activities.WALK_BACK: 7,
  Activities.WALK_QUICK: 8,
  Activities.WALK_STEP: 9,
  Activities.ELEVATOR: 10,
  Activities.TELEPHONE: 11,
}

def Index(e: Activities):
  return _indexDictionary[e]