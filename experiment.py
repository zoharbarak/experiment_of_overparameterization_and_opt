import random, datetime, math
import matplotlib.pyplot as plt
import numpy as np

def L(w, y):
    return 0.5 * ((w - y) ** 2)

def gradient(w, y, N):
    return N * (w - y) * math.pow(w, 2 - (2/float(N)))

def GD_step(w, y, step_size, N):
    return w - step_size*gradient(w, y, N)

def run_GD(w_0, y, step_size, N, max_iterations):
    losses = []
    w_values = []
    w = w_0
    for t in range(max_iterations):
        losses.append(L(w, y))
        w_values.append(w)
        w = GD_step(w, y, step_size, N)
    return losses, w_values

def plot_losses(losses, N_values):
    fig = plt.figure()
    for i, losses_lst in enumerate(losses):
        plt.plot(np.array(losses_lst))
    plt.legend(N_values, loc='best')
    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.title('Losses over time')
    plt.show()

def run_GD_on_net_depth(N, step_size, max_iterations=int(1e5)):
    w_0 = random.gauss(1, 1e-3)
    y = 100
    N_losses, w_values = run_GD(w_0, y, step_size=step_size, N=N, max_iterations=max_iterations)
    return N_losses

def plot_log_scale_loss(loss):
    fig = plt.figure()
    plt.plot(np.array(loss))
    plt.yscale('log')
    plt.xlabel('Time')
    plt.ylabel('Loss L (log scale)')
    plt.title('Losses for N=2 over time')
    plt.show()

def main():

    # Example - show the effects of network depths N
    print("[{}] Example - Overparameterization effects".format(datetime.datetime.now()))
    losses = []
    network_depths = [2, 3, 4, 8, 16]
    for N in network_depths:
        print("N = {}".format(N))
        losses.append(run_GD_on_net_depth(N=N, step_size=1e-7))
    plot_losses(losses ,network_depths)

    # Counter example - show the importance of choosing a small enough step size
    print("[{}] Counter example - too big step size effect".format(datetime.datetime.now()))
    plot_log_scale_loss(run_GD_on_net_depth(N=2, step_size=0.02, max_iterations=int(15)))
    print("[{}] Done".format(datetime.datetime.now()))


if __name__ == "__main__":
    main()
