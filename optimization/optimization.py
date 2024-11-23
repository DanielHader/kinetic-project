from matplotlib.tri import Triangulation, UniformTriRefiner
import matplotlib.pyplot as plt
import numpy as np
import heapq

def objective(fs, xs):
    return np.sum([f(x) for f, x in zip(fs, xs)])

class Simplex_PQNode:
    def __init__(self, points, fs, depth=0):
        # points is an NxN matrix (np.ndarray) where
        # rows represent points of the simplex
        # it is assumed that rows sum to 1
        self.points = points
        self.fs = fs

        self.depth = depth

        self._lb = None
        self._ub = None
        self._obj_vals = None
    
    def upper_bound(self):
        if self._ub is None:
            max_coords = np.max(self.points, axis=1)
            self._ub = objective(self.fs, max_coords)
        return self._ub

    def lower_bound(self):
        if self._lb is None:
            min_coords = np.min(self.points, axis=1)
            self._lb = objective(self.fs, min_coords)
        return self._lb

    def objective_values(self):
        if self._obj_vals is None:
            self._obj_vals = [objective(fs, point) for point in self.points]
        return self._obj_vals

    def __str__(self):
        return f'{self.points}\n{self.lower_bound()} -- {self.upper_bound()}'
    
    # comparison overloads for use as priority queue nodes
    # we want to prioritize nodes with larger lower bounds
    def __lt__(self, other):
        return self.depth < other.depth
    #return self.lower_bound() > other.lower_bound()

    def __eq__(self, other):
        return self.depth == other.depth
    #return self.lower_bound() == other.lower_bound()
    
    # split simplex along edge with biggest difference of objective function values
    def split(self):
        max_idx = np.argmax(self.objective_values())
        min_idx = np.argmin(self.objective_values())

        if min_idx == max_idx:
            min_idx = (min_idx + 1) % len(self.fs)

        max_point = self.points[max_idx]
        min_point = self.points[min_idx]

        avg_point = (max_point + min_point) / 2

        new_points1 = np.copy(self.points)
        new_points1[min_idx] = avg_point
        
        new_points2 = np.copy(self.points)
        new_points2[max_idx] = avg_point
        
        return [
            Simplex_PQNode(new_points1, fs, self.depth+1),
            Simplex_PQNode(new_points2, fs, self.depth+1),
        ]
        

# branch and bound approach
def maximize(fs, max_depth=15):
    N = len(fs)
    simplex_node = Simplex_PQNode(np.identity(N), fs)

    priority_queue = []
    heapq.heappush(priority_queue, simplex_node)

    biggest_value = float('-inf')
    best_u = None
    
    while len(priority_queue) > 0:
        node = heapq.heappop(priority_queue)

        if node.upper_bound() < biggest_value:
            continue
        if node.depth > max_depth:
            continue

        for i, val in enumerate(node.objective_values()):
            if val > biggest_value:
                biggest_value = val
                best_u = node.points[i]
                print(f'found new best value {biggest_value} at {best_u}')
                
        for child in node.split():
            heapq.heappush(priority_queue, child)

    return best_u, biggest_value
    
def plot_objective(fs, u_opt):
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

    N = 100
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

    x_opt = np.dot(tri_xs, u_opt)
    y_opt = np.dot(tri_ys, u_opt)

    ax.scatter([x_opt], [y_opt], c='red', s=100, marker='*')
    
    plt.colorbar(tcf)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    fs = [
        lambda x: 2 * x**(1/3),
        lambda x: 5 * x**2,
        lambda x: 0 if x <= 0.5 else x + 1
    ]

    u_opt, opt_val = maximize(fs)
    print(f'maximum value {opt_val} found at coordinates {u_opt}')
    plot_objective(fs, u_opt)
