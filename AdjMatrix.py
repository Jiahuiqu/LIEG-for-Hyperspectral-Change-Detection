import torch

a = torch.rand(1, 305, 448)


def L2Norm(data):
    out = torch.mul(data, data)
    out = torch.sqrt(torch.sum(out, 2))
    return out


def AdjMatrix(data):
    L2 = L2Norm(data)
    _, x, y = data.shape
    data = torch.reshape(data, [x, y])
    c = torch.mm(data, data.T)
    b = torch.mm(L2.T, L2)
    out = torch.div(c, b)
    print(out.shape)
    _, pred = out.topk(k=10, dim=1)
    out = torch.abs(out) * float('-inf')
    node, _ = out.shape
    for i in range(node):
        out[i, pred[i, :]] = 0
    return out


if __name__ == '__main__':
    out = AdjMatrix(a)
    print(out)
