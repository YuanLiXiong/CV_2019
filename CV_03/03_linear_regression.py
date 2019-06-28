# encoding=utf-8

from numpy import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

SEED = 1000
random.seed(SEED)

def inference(X, w, b):

    pred_Y = w * X + b

    return pred_Y


def eval_loss(X, Y, w, b):

    pred_Y = inference(X, w, b)
    d = pred_Y - Y
    loss = 0.5 * sum(power(d, 2)) / pred_Y.size

    return loss

def grad(pred_y, gt_y, x):
    diff = pred_y - gt_y
    dw = diff * x
    db = diff
    return dw, db


def cal_step_grad(X, gt_Y, w, b, batch_size, lr):
    pred_Y = inference(X, w, b)
    W, B = grad(pred_Y, gt_Y, X)
    dw = sum(W) / (W.size * batch_size)
    db = sum(B) / (B.size * batch_size)
    w -= dw * lr
    b -= db * lr
    return pred_Y, w, b


def train(X, gt_Y, batch_size, lr, max_iter):
    losses = []
    show_X, show_Y, show_GT_Y = [], [], []
    w, b = 0, 0

    for i in range(max_iter):
        train_X = sort(random.choice(X, batch_size))
        train_Y = sort(random.choice(gt_Y, batch_size))
        pred_Y, w, b = cal_step_grad(train_X, train_Y, w, b, batch_size, lr)

        show_X.append(train_X)
        show_GT_Y.append(train_Y)
        show_Y.append(pred_Y)

        loss = eval_loss(train_X, train_Y, w, b)
        losses.append(loss)

    show_loss(losses)
    show_line(show_X, show_GT_Y, show_Y, max_iter)
    # print("pred_w:", w, "  pred_b:", b, "  loss:", loss)

    return w, b, losses


def gen_sample_data(num_samples=100):
    w = random.randint(0, 10) + random.random()
    b = random.randint(0, 5) + random.random()

    X = random.rand(num_samples) * 100
    Y = w * X + b + random.random()

    return X, Y, w, b


def show_loss(losses):
    fig, ax = plt.subplots()
    t = 1000  # 显示时间间隔 1秒
    ax.set_title("loss")
    xdata, ydata = [], []
    ln, = ax.plot([], [], 'r-', animated=False)
    def init():
        ax.set_xlim(0, len(losses))
        ax.set_ylim(0, 2500)
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
    t = 1000
    ax2 = ax.twinx()
    ax.set_title("line")

    ln, = ax.plot([], [], 'b*', animated=False)
    ln2, = ax.plot([], [], 'r-', animated=False)

    def init():
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1000)
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 1000)
        return ln, ln2

    def update(n):
        xdata = X[n]
        pred_y = pred_Y[n]
        ydata = Y[n]
        ln.set_data(xdata, ydata)
        ln2.set_data(xdata, pred_y)
        return ln, ln2

    ani = FuncAnimation(fig, update, frames=arange(100),
                        interval=t,init_func=init, blit=True)

    plt.show()

def run():
    X, gt_Y, w, b = gen_sample_data(1000000)
    print("w: {},  b: {}".format(w, b))
    max_iter = 100
    lr = 0.01
    batch_size = 50

    pred_w, pred_b, losses = train(X, gt_Y, batch_size, lr, max_iter)

    print("pred_w: {},  pred_b: {}".format(pred_w, pred_b))
    print("loss: \n", losses)


if __name__ == "__main__":
    run()

