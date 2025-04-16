from matplotlib.animation import FuncAnimation
from numpy.typing import ArrayLike, NDArray
import matplotlib.pyplot as plt
import numpy as np

def make_ani(sub_path:NDArray, sub_noisy_path:NDArray, filter_estimates:NDArray, 
             filter_preds:list, filter_resamples:list,
             X:NDArray, Y:NDArray, p:ArrayLike, u:ArrayLike, v:ArrayLike, 
             title, figsize, dpi, interval, path):
    """Function to make animation."""

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.contourf(X, Y, p, alpha=0.5, cmap='turbo', levels=20, zorder=1)
    contour = ax.contour(X, Y, p, cmap='turbo', levels=10, zorder=2)
    ax.clabel(contour, inline=False, fontsize=12, colors = 'gray', zorder=3)
    ax.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2], zorder=4)

    total_n = sub_path.shape[0]
    num_obs = sub_noisy_path.shape[0]
    num_ests = filter_estimates.shape[0]
    num_res = filter_resamples.shape[0]
    obs_interval = total_n // num_obs 
    est_interval = total_n // num_ests
    res_interval = total_n // num_res

    sub_dot = ax.scatter([], [], color='blue', zorder=8)
    sub_line = ax.plot([], [], color='blue', label='Submarine Path', zorder=8)[0]
    sub_noisy = ax.scatter([], [], color='green', label='Noisy Submarine Position', zorder=6)
    est_line = ax.plot([], [], color='orange', label='Particle Filter Estimate', zorder=7)[0]
    est_dot = ax.scatter([], [], color='red', zorder=7)
    preds_dots = ax.scatter([], [], color='gray', s=2, label='Particle Prediction', zorder=5)
    res_dots = ax.scatter([], [], color='purple', s=2, label='Particle Resampling', zorder=7)

    ax.legend(loc='lower left')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    def update(idx, preds, resamples):
        sub_line.set_data(sub_path[:idx, 0], sub_path[:idx, 1])
        sub_dot.set_offsets([sub_path[idx]])

        particles = preds[idx]
        preds_dots.set_offsets(particles)

        if idx % obs_interval == 0:
            noisy_idx = idx // obs_interval
            sub_noisy.set_offsets(sub_noisy_path[:noisy_idx])

        if idx % est_interval == 0:
            est_idx = idx // est_interval
            est_line.set_data(filter_estimates[:est_idx, 0], filter_estimates[:est_idx, 1])
            est_dot.set_offsets([filter_estimates[est_idx]])

        if idx % res_interval == 0:
            res_idx = idx // res_interval
            parts = resamples[res_idx]
            res_dots.set_offsets(parts)

        return preds_dots, res_dots


    animate = FuncAnimation(fig=fig, func=update, frames=range(len(sub_path)), interval=interval, fargs=(filter_preds, filter_resamples))
    animate.save(path)
    plt.close()

def plotter(sub_pos:list, sub_noisy_pos:list, filter_estimates:list, filter_predictions:list, 
             filter_resamplings:list,
             X:NDArray, Y:NDArray, p:ArrayLike, u:ArrayLike, v:ArrayLike,
             title:str='Particle Filter on Submarine Trajectory',
             figsize:tuple[int]=(11, 7), dpi:int=250, animate:bool=False,
             animation_path:str="particle_ani.mp4", animation_interval:int=35):
    """Plot the system dynamics and animate if stated."""
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    cf = plt.contourf(X, Y, p, alpha=0.5, cmap='turbo', levels=20)
    contour = plt.contour(X, Y, p, cmap='turbo', levels=10)
    plt.clabel(contour, inline=False, fontsize=12, colors = 'black')

    # Quiver plot for velocity field
    quiv = plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2]) 
    plt.plot(*zip(*sub_pos), color='blue', label='Submarine path')
    plt.scatter(*zip(*sub_noisy_pos), color='red', s=10, label='Noisy Submarine Position')
    plt.plot(*zip(*filter_estimates), color='orange', label='Particle Filter estimate')
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower left')
    plt.show()

    sub_path = np.array(sub_pos)
    sub_noisy = np.squeeze(np.array(sub_noisy_pos))
    filter_est = np.array(filter_estimates)
    filter_resamplings = np.array(filter_resamplings)

    if animate:
        make_ani(sub_path, sub_noisy, filter_est, filter_predictions, filter_resamplings, X=X, Y=Y, p=p, u=u, v=v, title=title, 
                 figsize=figsize, dpi=dpi, path=animation_path, interval=animation_interval)