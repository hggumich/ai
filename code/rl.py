import numpy as np

ssp = [1, 1, 1, 1, 0]

asp = [1, 0]

def epoch():
    tr = 0
    for _ in range(100):
        a = np.random.choice(asp)
        s = np.random.choice(ssp)
        if a == s:
            tr += 1
    return tr

rl = np.array([epoch() for _ in range(15)])
print(rl)
print(rl.mean())

ssp = [1, 1, 1, 1, 0]

def epoch():
    tr = 0
    asp = [0, 1]
    for _ in range(100):
        a = np.random.choice(asp)
        s = np.random.choice(ssp)
        if a == s:
            tr += 1
        asp.append(s)
    return tr

rl2 = np.array([epoch() for _ in range(15)])
print(rl2)
print(rl2.mean())