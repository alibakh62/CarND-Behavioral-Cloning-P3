NVIDIA_H, NVIDIA_W = 200, 66

CONFIG = {
    'batchsize': 512,
    'input_width': 200,
    'input_height': 66,
    'input_channels': 3,
    'delta_correction': 0.25,
    'augmentation_steer_sigma': 0.2,
    'augmentation_value_min': 0.2,
    'augmentation_value_max': 1.5,
    'bias': 0.8,
    'crop_height': range(20, 140)
}