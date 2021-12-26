import torch

dtype = torch.float
device = torch.device("cpu")


def l1(y, y_pred, w):
    return (w * (y_pred - y)).abs().pow(1.1).mean()


def l1_d_reg(y_pred, x):
    dif_y = y_pred[1:] - y_pred[:-1]
    dif_x = x[1:] - x[:-1]
    steigung = dif_y / dif_x
    return steigung.abs().pow(1.1).mean()


def l2_dd_reg(y_pred, x):
    dif_y = y_pred[1:-1] - y_pred[2:] - y_pred[:-2]
    dif_x = x[2:] - x[:-2]
    steigung = dif_y / dif_x
    return steigung.abs().pow(1.1).mean()


def model(measurement, timestamp, weights, rs, lr, model="l1", reg="l2_dd"):
    x = torch.tensor(timestamp, device=device, dtype=dtype)
    y = torch.tensor(measurement, device=device, dtype=dtype)
    w = torch.tensor(weights, device=device, dtype=dtype)

    y_pred = torch.tensor(y, device=device, dtype=dtype, requires_grad=True)

    history = []
    for i in range(10001):
        loss = l1(y, y_pred, w) + rs * l2_dd_reg(y_pred, x)
        if i % 100 == 0:
            print(i, loss.item())
            history.append(loss.item())
        loss.backward()
        with torch.no_grad():
            y_pred -= lr * y_pred.grad
            y_pred.grad = None

    return y_pred, history


def model_2(measurement, timestamp, weights, rs, lr, model="l1", reg="l2_dd"):
    x = torch.tensor(timestamp, device=device, dtype=dtype)
    y = torch.tensor(measurement, device=device, dtype=dtype)
    w = torch.tensor(weights, device=device, dtype=dtype)

    y_pred = torch.tensor(y, device=device, dtype=dtype, requires_grad=True)

    maeLoss = torch.nn.L1Loss()

    d_y_l = (y_pred[0:-2] - y_pred[1:-1]) / (x[0:-2] - x[1:-1])
    d_y_r = y_pred[1:-1] - y_pred[2:] / (x[1:-1] - x[2:])
    dd_y = (d_y_l - d_y_r) / (0.5 * (x[0:-2] + x[1:-1]) - 0.5 * (x[1:-1] + x[2:]))

    grad = ((w * (y_pred - y)).abs().pow(1.1).mean() + rs * dd_y.abs().pow(1.1).mean()).backward()

    opt = torch.optim.RMSprop([y_pred], lr=lr)

    for i in range(10001):
        def closure():
            opt.zero_grad()
            output = y_pred
            loss = l1(y, y_pred, w) + rs * l2_dd_reg(y_pred, x)
            if i % 100 == 0:
                print(i, loss.item())
            loss.backward()
            return loss

        opt.step(closure)

# y_pred_final = y_pred.detach().numpy() * measurements_std + measurements_mean
# y_final = y.detach().numpy() * measurements_std + measurements_mean
