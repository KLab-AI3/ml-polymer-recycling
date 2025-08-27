import numpy as np
from utils.preprocessing import preprocess_spectrum, TARGET_LENGTH

def test_shapes_and_monotonicity():
    x = np.linspace(100, 200, 300)
    y = np.sin(x/10.0) + 0.01*(x - 100)
    x2, y2 = preprocess_spectrum(x, y, target_len=TARGET_LENGTH)
    assert x2.shape == (TARGET_LENGTH,)
    assert y2.shape == (TARGET_LENGTH,)
    assert np.all(np.diff(x2) > 0)

def test_idempotency():
    x = np.linspace(0, 100, 400)
    y = np.cos(x/7.0) + 0.002*x
    _, y1 = preprocess_spectrum(x, y, target_len=TARGET_LENGTH)
    _, y2 = preprocess_spectrum(np.linspace(x.min(), x.max(), TARGET_LENGTH), y1, target_len=TARGET_LENGTH)
    np.testing.assert_allclose(y1, y2, rtol=1e-6, atol=1e-7)
