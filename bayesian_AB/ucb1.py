import matplotlib.pyplot as plt
import numpy as np

# Also known as Multi Armed Bandit
NUM_TRIALS = 10000
EPS = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class Bandit:
    def __init__(self, p):
        # p is the win rate
        self.p = p
        self.p_estimate = 0.
        self.N = 0.

    def pull(self):
        # draw a 1 with probability p
        return np.random.random() < self.p

    def update(self, x):
        self.N = self.N + 1
        self.p_estimate = self.p_estimate + 1 / self.N * (x - self.p_estimate)


def ucb(mean, n, nj):
    return mean + np.sqrt(2 * np.log(n) / nj)


def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    rewards = np.zeros(NUM_TRIALS)
    total_plays = 0

    # initialize by playing each bandit
    for j in range(len(bandits)):
        x = bandits[j].pull()
        total_plays += 1
        bandits[j].update(x)

    for i in range(NUM_TRIALS):
        # use optimistic initial values to select the next bandit
        j = np.argmax([ucb(b.p_estimate, total_plays, b.N) for b in bandits])

        # pull the arm of bandit with largest sample
        x = bandits[j].pull()
        total_plays += 1

        # update rewards
        rewards[i] = x

        # update bandit
        bandits[j].update(x)

    cumulative_average = np.cumsum(rewards) / (np.arange(NUM_TRIALS) + 1)

    # plot moving average ctr
    plt.plot(cumulative_average)
    plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES))
    plt.xscale('log')
    plt.show()

    # plot moving average ctr linear
    plt.plot(cumulative_average)
    plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES))
    plt.show()

    for b in bandits:
        print(b.p_estimate)

    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    print("num times selected each bandit:", [b.N for b in bandits])

    return cumulative_average

if __name__ == "__main__":
    experiment()
