V = {'L1':0, 'L2':0.0}
new_V = V.copy()

cnt = 0 #用于计数
while True:
    new_V['L1'] = 0.5 *(-1+0.9*V['L1']+0.5*(1+0.9*V['L2']))
    new_V['L2'] = 0.5*(0+0.9*V['L1']+0.5*(-1+0.9*V['L2']))

    #更行量的最大值
    delta = abs(new_V['L1']-V['L1'])
    delat = max(delta, abs(new_V['L2']-V['L2']))
    V = new_V.copy()

    cnt += 1

    if delta <1e-4:
        print(V)
        print(cnt)
        break