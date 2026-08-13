import numpy as np
import cv2

def make_phantom(size=256, tumour=True, tumour_side="right", seed=0,
                 bias=True, noise=True):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2

    cv2.ellipse(img, (cx, cy), (int(size * 0.42), int(size * 0.46)),
                0, 0, 360, 210, -1)
    cv2.ellipse(img, (cx, cy), (int(size * 0.37), int(size * 0.41)),
                0, 0, 360, 120, -1)
    for sign in (-1, 1):
        cv2.ellipse(img, (cx + sign * int(size * 0.07), cy),
                    (int(size * 0.035), int(size * 0.10)), 0, 0, 360, 60, -1)
    texture = rng.normal(0, 6, (size, size)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (0, 0), 2.0)
    brain_region = np.zeros((size, size), np.uint8)
    cv2.ellipse(brain_region, (cx, cy), (int(size * 0.37), int(size * 0.41)),
                0, 0, 360, 1, -1)
    img += texture * brain_region

    centre = None
    if tumour:
        offset = int(size * 0.15) * (1 if tumour_side == "right" else -1)
        centre = (cx + offset, cy - int(size * 0.08))
        cv2.circle(img, centre, int(size * 0.075), 195, -1)
        cv2.circle(img, centre, int(size * 0.040), 235, -1)

    if bias:
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        field = 0.70 + 0.60 * (xx / size)
        img = img * field

    if noise:
        img = img + rng.normal(0, 4, (size, size)).astype(np.float32)

    return np.clip(img, 0, 255).astype(np.uint8), centre
