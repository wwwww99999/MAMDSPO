def get_velocity_threshold(env_name):
    thresholds = {'Safety2x3HalfCheetahVelocity-v0': 3.227,
                  'Safety3x1HopperVelocity-v0': 0.9613}
    return thresholds[env_name]