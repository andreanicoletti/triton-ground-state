import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

class EulerIntegrator:

    def __init__(self, A, dt):
        self.A = A
        self.dt = dt

    def step(self, x, a, t) -> tuple[NDArray, NDArray, float]:
        t_new = t + self.dt
        x_new = x + self.dt * a
        a_new = self.A(x_new, t_new)
        return x_new, a_new, t_new # we also return the acceleration for more efficient computations

    def evolve(self, x0, n_steps, checkpoints=None, verbose=False, init_msg=None, end_msg=None) -> tuple[NDArray, NDArray]:
        checkpoints = checkpoints if checkpoints is not None else n_steps
        
        # save_interval = max(1, n_steps // (checkpoints - 1)) # one checkpoint is already at the initial time
        
        save_indices = np.linspace(1, n_steps, checkpoints, dtype=int)
        if verbose:
            print("checkpoints:")
            print(save_indices)

        pos = np.zeros((checkpoints+1, *x0.shape))
        times = np.zeros(checkpoints+1)

        pos[0] = x = x0
        times[0] = t = 0.0
        a = self.A(x0, 0.0)

        if init_msg is not None:
            print(init_msg)

        k = 1
        for i in tqdm(range(1, n_steps+1), disable=not verbose):
            x, a, t = self.step(x, a, t)

            if i in save_indices:
                pos[k] = x
                times[k] = t
                k += 1

        if end_msg is not None:
            print(end_msg)

        return pos, times
