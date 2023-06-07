from demand_dists import *

configs = {
    "config1": {
        "p": 1,
        "c_h": 1,
        "c_s": 1,
        "T": 1_000,
        "W": Binomial(100, 0.1),
    },
    "config2": {
        "p": 1,
        "c_h": 1,
        "c_s": 1,
        "T": 1_000,
        "W": Binomial(1000, 0.1),
    },
    "config3": {
        "p": 1,
        "c_h": 1,
        "c_s": 1,
        "T": 1_000,
        "W": WeekendHeavy(n1=1000, p1=0.1, n2=1000, p2=0.1),
    },
    "config4": {
        "p": 1,
        "c_h": 1,
        "c_s": 1,
        "T": 1_000,
        "W": IncreasingDemand(1000, 0.1, 0.1)
    },
}