import numpy as np
import source.tiles3 as tc

class PendulumTileCoder:
    def __init__(self, iht_size=4096, num_tilings=32, num_tiles=8):
        """
        Initializes the MountainCar Tile Coder
        Initializers:
        iht_size -- int, the size of the index hash table, typically a power of 2
        num_tilings -- int, the number of tilings
        num_tiles -- int, the number of tiles. Here both the width and height of the tiles are the same

        Class Variables:
        self.iht -- tc.IHT, the index hash table that the tile coder will use
        self.num_tilings -- int, the number of tilings the tile coder will use
        self.num_tiles -- int, the number of tiles the tile coder will use
        """

        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        self.iht = tc.IHT(iht_size)

    def get_tiles(self, angle, ang_vel):
        """
        Takes in an angle and angular velocity from the pendulum environment
        and returns a numpy array of active tiles.

        Arguments:
        angle -- float, the angle of the pendulum between -np.pi and np.pi
        ang_vel -- float, the angular velocity of the agent between -2*np.pi and 2*np.pi

        returns:
        tiles -- np.array, active tiles

        """

        ### Use the ranges above and scale the angle and angular velocity between [0, 1]
        # then multiply by the number of tiles so they are scaled between [0, self.num_tiles]

        angle_scaled = 0
        ang_vel_scaled = 0

        MIN_ANGLE, MAX_ANGLE = -np.pi, np.pi
        MIN_VEL, MAX_VEL = -2 * np.pi, 2 * np.pi

        angle_scale = self.num_tiles / np.abs(MAX_ANGLE - MIN_ANGLE)
        ang_vel_scale = self.num_tiles / np.abs(MAX_VEL - MIN_VEL)

        angle_scaled = angle_scale * (MIN_ANGLE + angle)
        ang_vel_scaled = ang_vel_scale * (MIN_VEL + ang_vel)

        # Get tiles by calling tc.tileswrap method
        # wrapwidths specify which dimension to wrap over and its wrapwidth
        tiles = tc.tileswrap(self.iht, self.num_tilings, [angle_scaled, ang_vel_scaled],
                             wrapwidths=[self.num_tiles, False])

        return np.array(tiles)