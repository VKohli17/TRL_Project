import numpy as np

class Gaussian:
    def __init__(self, mean, stdev, seed=42):
        self.mean = mean
        self.stdev = stdev
        self.prng = np.random.default_rng(seed)

    def __call__(self, t: int):
        """
        Since this is iid Gaussian, we don't need to use the time
        :param t: time
        :return: demand"""
        return self.prng.normal(self.mean, self.stdev)

    def __str__(self):
        return f'N({self.mean}, {self.stdev})'


class Binomial:
    def __init__(self, n, p, seed=42):
        self.n = n
        self.p = p
        self.mean = n*p
        self.std = np.sqrt(n * p * (1 - p))
        self.prng = np.random.default_rng(seed)

    def __call__(self, t: int):
        """
        Since this is iid Binomial, we don't need to use the time
        :param t: time
        :return: demand"""
        return int(self.prng.binomial(self.n, self.p))

    def __str__(self):
        return f'B({self.n}, {self.p})'
    

class Uniform:
    def __init__(self, a, b, seed=42):
        self.a = a
        self.b = b
        self.prng = np.random.default_rng(seed)

    def __call__(self, t: int):
        """
        Since this is iid Uniform, we don't need to use the time
        :param t: time
        :return: demand"""
        return self.prng.uniform(self.a, self.b)

    def __str__(self):
        return f'U({self.a}, {self.b})'


class WeekendHeavy:
    def __init__(self, n1, p1, n2, p2, seed=42) -> None:
        """
        Binomial Distributions
        :param n1: n for weekend
        :param p1: p for weekend
        :param n2: n for weekday
        :param p2: p for weekday
        """
        self.n1 = n1
        self.p1 = p1
        self.n2 = n2
        self.p2 = p2
        self.prng = np.random.default_rng(seed)
    
    def __call__(self, t: int):
        if t % 7 == 5 or t % 7 == 6:
            return int(self.prng.binomial(self.n1, self.p1))
        else:
            return int(self.prng.binomial(self.n2, self.p2))
    
    def __str__(self):
        return f'Weekend - Binomiall({self.n1}, {self.p1}), Weekday - Binomial({self.n2}, {self.p2})'


class IncreasingDemand:
    def __init__(self, n, p, rate, seed=42) -> None:
        self.n = n
        self.p = p
        self.rate = rate
        self.prng = np.random.default_rng(seed)
    
    def __call__(self, t: int):
        self.n = self.n * (1 + self.rate)
        return int(self.prng.binomial(int(self.n), self.p))
    
    def __str__(self):
        return f'Increasing Demand with rate {self.rate}- Binomial({self.n}, {self.p})'