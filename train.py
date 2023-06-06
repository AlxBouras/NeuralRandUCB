import numpy as np
from tqdm import tqdm


def train(bandit, learner, T=6000):
    regrets = []
    for t in tqdm(range(T)):
        context, rwd = bandit.step()
        arm_select, nrm, sig, ave_rwd = learner.select(context)
        r = rwd[arm_select]
        reg = np.max(rwd) - r
        loss = learner.train(context[arm_select], r)
        regrets.append(reg)
        if t % 100 == 0:
            print('{}: {:.3f}, {:.3e}, {:.3e}, {:.3e}, {:.3e}'.format(t, np.sum(regrets), loss, nrm, sig, ave_rwd))

    return np.cumsum(regrets)
