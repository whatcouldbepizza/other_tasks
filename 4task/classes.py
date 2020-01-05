from matplotlib.patches import Circle

import math


class Particle:
    """
    Class describing a single particle
    """
    def __init__(self, coordinates=[0, 0], speed=[0.0, 0.0], mass=1, color="red", living_time=100000):
        """
        Main constructor
        """
        self.coordinates = coordinates
        self.speed = speed
        self.mass = mass
        self.color = color
        self.living_time = living_time
        try:
            self.circle = Circle((coordinates[0], coordinates[1]), self.mass, color=self.color)
        except Exception:
            print("Failed to create circle with entered color, creating red instead...")
            self.circle = Circle((coordinates[0], coordinates[1]), self.mass, color="red")

    def create_circle(self, coordinates, size, color="red"):

        self.circle = Circle((coordinates[0], coordinates[1]), size, color=color)

    def __str__(self):
        return "Particle information:\n    coordinates: [{},{}]".format(self.coordinates[0], self.coordinates[1]) + \
               "\n    speed: [{},{}]".format(self.speed[0], self.speed[1]) + \
               "\n    mass: {}".format(self.mass) + \
               "\n    color: {}".format(self.color) + \
               "\n    living_time: {}".format(self.living_time) + \
               "\n    cirlcle: {}".format(self.circle)


class Emitter:
    """
    Class describing emitter
    """
    def __init__(self, coordinates=[1.0, 1.0], emitting_vector=[1, 1]):
        """
        Main constructor
        """
        self.coordinates = coordinates
        self.vector = emitting_vector

    def change_position(self, coordinates, vector):
        """
        Function to change position of emitting source
        """
        self.coordinates = coordinates
        self.vector = vector
