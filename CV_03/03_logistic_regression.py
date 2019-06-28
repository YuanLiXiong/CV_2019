# encoding=utf-8

from numpy import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

SEED = 1000
random.seed(SEED)

def inference(X, sita):

    pred_Y = 1 / (exp(-sita * X) + 1)

    return pred_Y


def eval_loss(X, Y, sita):

    pred_Y = inference(X, sita)
    d = pred_Y - Y
    loss = 0.5 * sum(power(d, 2)) / pred_Y.size

    return loss

def grad(pred_y, gt_y, x):
    diff = pred_y - gt_y
    d_sita = diff * x
    return d_sita


def cal_step_grad(X, gt_Y, sita, batch_size, lr):
    pred_Y = inference(X, sita)
    grad_sita = grad(pred_Y, gt_Y, X)
    diff_sita = sum(grad_sita) / (grad_sita.size * batch_size)
    sita -= diff_sita * lr
    return pred_Y, sita


def train(X, gt_Y, batch_size, lr, max_iter):
    losses = []
    show_X, show_Y, show_GT_Y = [], [], []
    sita = 0

    for i in range(max_iter):
        train_X = X
        train_Y = gt_Y
        pred_Y, sita = cal_step_grad(train_X, train_Y, sita, batch_size, lr)

        show_X.append(train_X)
        show_GT_Y.append(train_Y)
        show_Y.append(pred_Y)

        loss = eval_loss(train_X, train_Y, sita)
        losses.append(loss)

    show_loss(losses)
    show_line(show_X, show_GT_Y, show_Y, max_iter)
    print("sita:", sita, "  loss:", loss)

    return sita, losses


def gen_sample_data(num_samples=200):

    sita = 1

    # X range: [-100, 100]
    X = random.rand(num_samples) * num_samples - num_samples // 2
    X = sort(X)
    Y = zeros_like(X)
    Y[X > 0] = 1
    return X, Y, sita


def show_loss(losses):
    fig, ax = plt.subplots()
    t = 1000  # 显示时间间隔 1秒
    ax.set_title("loss")
    xdata, ydata = [], []
    ln, = ax.plot([], [], 'r-', animated=False)
    def init():
        ax.set_xlim(0, len(losses))
        ax.set_ylim(0, 1)
        return ln,
    def update(n):
        xdata.append(n)
        ydata.append(losses[n])
        ln.set_data(xdata, ydata)
        return ln,

    ani = FuncAnimation(fig, update, frames=arange(len(losses)),
                        init_func=init, interval=t, blit=True)

    plt.show()


def show_line(X, Y, pred_Y, max_iter):
    fig, ax = plt.subplots()
    t = 100
    ax2 = ax.twinx()
    ax.set_title("line")

    ln, = ax.plot([], [], 'b*', animated=False)
    ln2, = ax.plot([], [], 'r-', animated=False)

    def init():
        ax.set_xlim(-100, 100)
        ax.set_ylim(0, 1)
        ax2.set_xlim(-100, 100)
        ax2.set_ylim(0, 1)
        return ln, ln2

    def update(n):
        xdata = X[n]
        pred_y = pred_Y[n]
        ydata = Y[n]
        ln.set_data(xdata, ydata)
        ln2.set_data(xdata, pred_y)
        return ln, ln2

    ani = FuncAnimation(fig, update, frames=arange(len(X)),
                        interval=t,init_func=init, blit=True)

    plt.show()

def run():
    X, gt_Y, sita = gen_sample_data(1000)
    print("sita: {}".format(sita))
    max_iter = 100
    lr = 0.01
    batch_size = 50

    sita, losses = train(X, gt_Y, batch_size, lr, max_iter)

    print("sita: {}".format(sita))
    print("loss: \n", losses)


if __name__ == "__main__":
    run()

