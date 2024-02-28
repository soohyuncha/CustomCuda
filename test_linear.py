import torch
import torch.nn.functional as F
import linear_custom

if __name__ == '__main__':
    M, K, N = 256, 1024, 256
    bs = 32

    x = torch.rand(M, K).type(torch.float32).to('cuda')
    W = torch.rand(N, K).type(torch.float32).to('cuda')

    y1 = F.linear(x, W)
    #print(y1)

    x = x.reshape(-1)
    W = W.reshape(-1)

    y2 = linear_custom.linear(x, W, M, N, K, bs)
    y2 = y2.reshape(M, N)
    #print(y2)
    print('calc done', flush=True)

    # Result check
    cnt = 0
    for i in range(M):
        for j in range(N):
            if abs(y2[i, j] - y1[i, j]) < 0.001:
                continue
            else:
                print(f'Error at [{i}, {j}]: {y1[i, j]: .5f}, {y2[i, j]: .5f}')
                cnt += 1
    if cnt == 0:
        print('All same')
