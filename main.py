import numpy as np

def mean_std(arr):
    """Повертає (mean, std) для списку або масиву."""
    a = np.array(arr, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))

def linear_regression(x, y):
    """
    Простий OLS для y = intercept + slope * x
    Повертає (intercept, slope, y_pred)
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    X = np.vstack([np.ones_like(x), x]).T  # design matrix
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, slope = float(coef[0]), float(coef[1])
    y_pred = intercept + slope * x
    return intercept, slope, y_pred

def demo():
    """Демонстраційний запуск — генерація даних + вивід."""
    print("DRONE TEST: numpy-stats-demo")
    # прості дані з шумом
    x = np.arange(0, 10)
    rng = np.random.default_rng(42)
    y = 2.5 * x + 1.0 + rng.normal(scale=0.5, size=x.shape)
    mu, sigma = mean_std(y)
    intercept, slope, _ = linear_regression(x, y)
    print(f"mean={mu:.3f}, std={sigma:.3f}")
    print(f"linear regression: intercept={intercept:.3f}, slope={slope:.3f}")

if __name__ == "__main__":
    demo()
