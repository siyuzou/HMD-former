import numpy as np


def random_uniform_disc():
    """Samples a random 2D point from the unit disc with a uniform distribution."""
    angle = np.random.uniform(-np.pi, np.pi)
    radius = np.sqrt(np.random.uniform(0, 1))
    return radius * np.array([np.cos(angle), np.sin(angle)])
