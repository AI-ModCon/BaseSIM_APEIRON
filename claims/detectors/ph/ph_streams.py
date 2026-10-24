import math
def streams():
    wobble = lambda i: 0.03 * math.sin(i * 0.7) + 0.01 * math.cos(i * 2.3)
    return {
        "step":        [(0.20 if i < 200 else 0.80) + wobble(i) for i in range(400)],
        "constant":    [0.5] * 200,
        "ramp":        [0.1 + 0.004 * i for i in range(250)],
        "spike":       [0.3] * 100 + [9.0] + [0.3] * 99,
        "down_step":   [(0.80 if i < 200 else 0.20) + wobble(i) for i in range(400)],
        "down_ramp":   [0.9 - 0.004 * i for i in range(250)],
        # dip-and-recover: the regime that separates the clamped form from the
        # running-minimum form, found by held-out probe
        "slow_sine":   [0.5 + 0.3 * math.sin(i / 60) for i in range(400)],
        "slow_cosine": [0.5 + 0.25 * math.cos(i / 45) for i in range(400)],
        "dip_return":  [0.6] * 100 + [0.2] * 120 + [0.6] * 180,
        "rise_return": [0.3] * 100 + [0.75] * 120 + [0.3] * 180,
        "v_shape":     [0.8 - 0.006 * i for i in range(100)] + [0.2 + 0.006 * i for i in range(150)],
        "two_regimes": [0.2] * 150 + [0.7] * 150 + [0.35] * 150,
    }
