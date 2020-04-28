import noise
from matplotlib import pyplot as plt
import numpy as np

import timeit

class Gridmap:
    def __init__(
        self,
        xlim,
        ylim,
        occupancy_grid
    ):
        self.xlim = xlim
        self.ylim = ylim

        # Number of squares
        self.M = np.size(occupancy_grid, axis=0)
        self.N = np.size(occupancy_grid, axis=1)

        self.xres = (xlim[1] - xlim[0]) / self.M
        self.yres = (ylim[1] - ylim[0]) / self.N

        self.xaxis = np.linspace(xlim[0], xlim[1], self.M + 1)
        self.yaxis = np.linspace(ylim[0], ylim[1], self.N + 1)

        self.og = occupancy_grid

        self.bm = {
            "value_at": 0,
            "value_at_i": 0,
            "clear_view": 0
        }

    def plot(self, plot_axes=None):
        if plot_axes is None:
            fig = plt.figure()
            plot_axes = fig.add_subplot(111)

        plot_axes.pcolormesh(
            self.xaxis,
            self.yaxis,
            self.og.T,
            vmin=0,
            vmax=1,
            cmap='gray_r',
            linewidths=0.1,
            edgecolor='k'
        )

    def value_at(self, position):
        tt0 = timeit.default_timer()
        i = ((position[:,0] - self.xaxis[0])/self.xres).astype(int)
        j = ((position[:,1] - self.yaxis[0])/self.yres).astype(int)

        self.bm["value_at"] += timeit.default_timer() - tt0
        self.bm["value_at_i"] += 1

        return self.og[i,j]

    def clear_view(self, p1, p2):
        a = (p2[1] - p1[1])/(p2[0] - p1[0])
        b = p1[1] - p1[0] * a

        line = lambda x: a * x + b
        invline = lambda y: (y - b) / a

        ax = plt.gca()
        ax.plot(p1[0], p1[1], 'ro')
        ax.plot(p2[0], p2[1], 'bo')

        xl = np.linspace(p1[0], p2[0], 3)
        yl = line(xl)
        ax.plot(xl, yl)

        xmin = min([p1[0], p2[0]])
        xmax = max([p1[0], p2[0]])
        ymin = min([p1[1], p2[1]])
        ymax = max([p1[1], p2[1]])

        xintersects = self.xaxis[(self.xaxis > xmin) & (self.xaxis < xmax)]
        xintersects = np.c_[xintersects, line(xintersects)]
        yintersects = self.yaxis[(self.yaxis > ymin) & (self.yaxis < ymax)]
        yintersects = np.c_[invline(yintersects), yintersects]

        if xintersects.size > 0:
            dx = np.array([self.xres / 2, 0])
            squares = np.r_[xintersects + dx, xintersects - dx]
            if self.value_at(squares).astype(bool).any():
                return False

        if yintersects.size > 0:
            dy = np.array([0, self.yres / 2])
            squares = np.r_[yintersects + dy, yintersects - dy]
            if self.value_at(squares).astype(bool).any():
                return False

        return True

    @classmethod
    def from_perlin_noise(
            cls,
            xlim,
            ylim,
            M,
            N,
            seed=4,
            scale=40,
            threshold=0.7
    ):

        pmap = cls.perlin_map(xlim, ylim, M, N, seed, scale)

        # Make it binary
        maximum = np.max(pmap)
        minimum = np.min(pmap)
        threshold = minimum + threshold * (maximum - minimum)

        pmap = (pmap > threshold).choose(pmap, 1)
        pmap = (pmap <= threshold).choose(pmap, 0)

        return cls(
            xlim=xlim,
            ylim=ylim,
            occupancy_grid=pmap
        )

    @staticmethod
    def perlin_map(xlim, ylim, M, N, seed=None, scale=40):
        if seed is None:
            seed = np.random.randint(0, 200)

        xres = (xlim[1] - xlim[0]) / M
        yres = (ylim[1] - ylim[0]) / N

        xaxis = np.linspace(xlim[0], xlim[1], M + 1)
        yaxis = np.linspace(ylim[0], ylim[1], N + 1)

        perlin_map = np.zeros((M, N))

        for i, x in enumerate(perlin_map):
            for j, y in enumerate(perlin_map[i]):
                perlin_map[i,j] = noise.pnoise2(
                    (xaxis[i] + xres/2) / scale,
                    (yaxis[j] + yres/2) / scale,
                    base=seed
                )

        return perlin_map

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)

    gm = Gridmap.from_perlin_noise(
        xlim=np.array([-50, 50]),
        ylim=np.array([-50, 50]),
        M=100,
        N=100,
        seed=6,
        scale=30,
        threshold=0.9
    )

    gm.plot(plot_axes=ax)

    p1 = 100 * np.random.rand(2) - 50
    p2 = 100 * np.random.rand(2) - 50

    tt = timeit.default_timer()
    clr = gm.clear_view(p1, p2)
    print(timeit.default_timer() - tt)

    plt.show()
