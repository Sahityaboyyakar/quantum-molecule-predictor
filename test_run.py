import numpy as np
from run import make_data, metrics


def test_make_data_shapes():
    X, y = make_data(n_samples=40, seed=42)
    assert X.shape == (40, 4)
    assert y.shape == (40,)
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()


def test_metrics_keys():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    out = metrics(y_true, y_pred)
    assert "rmse" in out
    assert "r2" in out
    assert out["rmse"] >= 0.0