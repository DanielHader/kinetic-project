from matplotlib.tri import Triangulation, UniformTriRefiner
import matplotlib.pyplot as plt
import numpy as np

def objective(fs, xs):
    return np.sum([f(x) for f, x in zip(fs, xs)])

class BarycentricSimplexDomain:
    def __init__(self, points, fs):
        # points is an NxN matrix where
        # rows represent points of the simplex
        # it is assumed that rows sum to 1
        self.points = points
        self.fs = fs

    def upper_bound(self):
        max_coords = np.max(self.points, axis=1)
        return objective(self.fs, max_coords)

    def lower_bound(self):
        min_coords = np.min(self.points, axis=1)
        return objective(self.fs, min_coords)

    # split simplex along edge with biggest difference of objective function values
    def split(self):
        obj_values = [objective(fs, point) for point in self.points]
        max_idx = np.argmax(obj_values)
        min_idx = np.argmin(obj_values)

    
    def maximize(self):
        pass

def plot_objective(fs):
    # barycentric triangle corners

    padding=0.1
    
    angles = [np.pi*(1/2 + 2/3*x) for x in range(3)]
    tri_xs = np.cos(angles)
    tri_ys = np.sin(angles)

    ax_lim_min = np.min([tri_xs, tri_ys], axis=1) - padding
    ax_lim_max = np.max([tri_xs, tri_ys], axis=1) + padding

    xs = []
    ys = []
    zs = []

    N = 200
    for s1 in np.arange(N+1):
        u1 = s1 / N
        for s2 in np.arange(N-s1+1):
            u2 = s2 / N
            u3 = 1 - u1 - u2

            # barycentric coords
            u = np.array([u1, u2, u3])
            
            x = np.dot(tri_xs, u)
            y = np.dot(tri_ys, u)
            z = fs[0](u1) + fs[1](u2) + fs[2](u3)
            
            xs.append(x)
            ys.append(y)
            zs.append(z)

            
            
    outer_tri = Triangulation(tri_xs, tri_ys)
    
    domain_tri = Triangulation(xs, ys)
    refiner = UniformTriRefiner(domain_tri)
    
    tri_refi, zs_refi = refiner.refine_field(zs, subdiv=1)

    fig, ax = plt.subplots()
    ax.set_xlim(ax_lim_min[0], ax_lim_max[0])
    ax.set_ylim(ax_lim_min[1], ax_lim_max[1])
    ax.set_aspect('equal')

    # ax.axis('off')
    ax.triplot(outer_tri)
    num_levels = 50
    min_z = np.min(zs)
    max_z = np.max(zs)
    
    levels = np.arange(min_z, max_z*1.1, (max_z - min_z) / num_levels)
    
    tcf = ax.tricontourf(tri_refi, zs_refi, levels=levels, cmap='terrain')
    tc = ax.tricontour(tri_refi, zs_refi, levels=levels,
                       colors=['0.25', '0.5', '0.5', '0.5', '0.5'],
                       linewidths=[1.0, 0.5, 0.5, 0.5, 0.5])

    # ax.scatter(xs, ys)

    plt.colorbar(tcf)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    fs = [
        lambda x: 2 * x**(1/3),
        lambda x: 5 * x**2,
        lambda x: 0 if x <= 0.5 else x + 1
    ]

    simplex = BarycentricSimplex(np.identity(3))

    # plot_objective(fs)
