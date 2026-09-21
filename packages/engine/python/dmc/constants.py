"""Shape constants for the DMC pipeline. Must match the TS-side encoders
in packages/engine/src/rl/dmc/{cardEncoding,observation,actions}.ts.
"""

# Card encoding (cardEncoding.ts).
ENCODING_LEN = 108  # 54 slots x 2 rows

# Observation (observation.ts).
HISTORY_MAX_STEPS = 100
STATIC_CHANNELS = 7
PER_STEP_DIM = ENCODING_LEN + 4 + 1  # cards + seat one-hot + trick-position
SCALAR_DIM = 16
STATIC_DIM = STATIC_CHANNELS * ENCODING_LEN  # 756

# Action encoding (actions.ts).
SHAPE_DIM = 5  # follow / lead-single / lead-pair / lead-tractor / lead-stop
ACTION_DIM = ENCODING_LEN + SHAPE_DIM + 2  # 115

# Roles (roles.ts).
ROLES = ('landlord', 'defender', 'attacker')

# DMC hyperparameters.
GAMMA = 0.95
EPS_GREEDY = 0.015
