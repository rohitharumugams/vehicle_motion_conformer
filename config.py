import numpy as np

DATA = '/Users/rohith/Desktop/Rohith/Doppler/DopplerSim_2.0/static/batch_outputs/vs13_path2d_10_speed_bins'
STATE = 'canonical_state_frames.npy'
NAMES = ('vx', 'vy', 'px', 'py')
BINS = [np.arange(-120, 122, 2, dtype=np.float32),
        np.arange(-120, 122, 2, dtype=np.float32),
        np.arange(-200, 202, 2, dtype=np.float32),
        np.arange(-100, 102, 2, dtype=np.float32)]
OUTPUT_HZ = 10
SPEC_SR = 22050
N_FFT = 2048
HOP = 256
FEATURES = N_FFT // 2 + 1
# Existing exported STFT is assumed centered; frames are delayed until available.
STFT_CENTERED = True

HIDDEN = 128
LAYERS = 4
ATTENTION_HEADS = 4
LEFT_CONTEXT = 1024  # Feature frames per layer, about 11.9 s at this STFT hop.
CONV_KERNEL = 15
POSITION_SCALE = 100.0
DROPOUT = 0.1
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-4
SEED = 42
HEAD_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Enable after verifying physically consistent labels (the supplied sample has
# abrupt velocity boundary changes). Four-head CE is the default baseline.
KINEMATIC_WEIGHT = 0.0
ACCELERATION_WEIGHT = 0.0
MAX_ACCEL_MPS2 = 8.0  # Tune to the simulator; vector acceleration, not per axis.
MAX_SPEED_KMPH = 120.0
BEAM_SIZE = 32
INITIAL_TOP_K = 8
BRANCH_TOP_K = 3
DECODER_ACCEL_PENALTY = 0.1


def model_config():
    return dict(feat_in=FEATURES, hidden=HIDDEN, layers=LAYERS,
                attention_heads=ATTENTION_HEADS, left_context=LEFT_CONTEXT,
                conv_kernel=CONV_KERNEL, position_scale=POSITION_SCALE,
                dropout=DROPOUT)


def feature_config():
    return dict(spec_sr=SPEC_SR, n_fft=N_FFT, hop=HOP,
                centered=STFT_CENTERED, output_hz=OUTPUT_HZ)
