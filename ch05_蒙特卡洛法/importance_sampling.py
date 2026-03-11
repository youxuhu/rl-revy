import numpy as np

# if __name__ == "__main__":
#     x = np.array([1,2,3])
#     pi = np.array([0.1, 0.1, 0.8])

#     #期望值
#     e = np.sum(x * pi)
#     print('E_pi[x]:', e)

#     #蒙特卡罗方法
#     n = 100
#     samples = []
#     for _ in range(n):
#         s = np.random.choice(x, p = pi)
#         samples.append(s)
    
#     mean = np.mean(samples)
#     var = np.var(samples)
#     print('MC : mean={:.2f}, var={:.2f}'.format(mean, var))


if __name__ == "__main__":
    x = np.array([1,2,3])
    pi = np.array([0.1, 0.1, 0.8])

    b = np.array([1/3, 1/3, 1/3])
    n = 100
    samples = []

    for _ in range(n):
        idx = np.arange(len(b))
        i = np.random.choice(idx, p = b)
        s = x[i]
        rho = pi[i] / b[i]
        samples.append(s * rho)
    
    mean = np.mean(samples)
    var = np.var(samples)
    print('IS : mean={:.2f}, var={:.2f}'.format(mean, var))
