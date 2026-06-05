import os
import sys

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from scene import get_camera_config, query_radiance_field


def generate_rays(H, W, focal, c2w):
    u = np.arange(W, dtype=np.float32) + 0.5
    v = np.arange(H, dtype=np.float32) + 0.5
    uu, vv = np.meshgrid(u, v, indexing="xy")

    camera_dirs = np.stack(
        [
            (uu - W * 0.5) / focal,
            -(vv - H * 0.5) / focal,
            np.ones_like(uu),
        ],
        axis=-1,
    )
    camera_dirs = camera_dirs / np.linalg.norm(camera_dirs, axis=-1, keepdims=True)

    rotation = c2w[:3, :3]
    rays_d = camera_dirs @ rotation.T
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape).copy()
    return rays_o.astype(np.float32), rays_d.astype(np.float32)


def sample_points(rays_o, rays_d, near, far, num_samples):
    t_vals = np.linspace(near, far, num_samples, dtype=np.float32)
    points = rays_o[..., None, :] + rays_d[..., None, :] * t_vals[None, None, :, None]

    if num_samples > 1:
        last_delta = t_vals[-1] - t_vals[-2]
    else:
        last_delta = far - near
    deltas = np.diff(t_vals, append=t_vals[-1] + last_delta).astype(np.float32)
    return points.astype(np.float32), t_vals, deltas


def volume_render(sigmas, rgbs, deltas, t_vals):
    deltas = np.asarray(deltas, dtype=np.float32)
    if deltas.ndim == 1:
        deltas = deltas[None, None, :]

    alphas = 1.0 - np.exp(-sigmas * deltas)
    transmittance = np.cumprod(
        np.concatenate(
            [np.ones_like(alphas[..., :1]), 1.0 - alphas + 1e-10],
            axis=-1,
        ),
        axis=-1,
    )[..., :-1]
    weights = transmittance * alphas

    pred_rgb = np.sum(weights[..., None] * rgbs, axis=-2)
    pred_depth = np.sum(weights * t_vals.reshape(1, 1, -1), axis=-1)
    return pred_rgb.astype(np.float32), pred_depth.astype(np.float32), weights.astype(np.float32)


def run(base_dir=None, results_dir=None):
    base_dir = os.path.dirname(os.path.abspath(__file__)) if base_dir is None else base_dir
    results_dir = (
        os.path.abspath(os.path.join(base_dir, "..", "results"))
        if results_dir is None
        else results_dir
    )
    os.makedirs(results_dir, exist_ok=True)

    cfg = get_camera_config()
    rays_o, rays_d = generate_rays(cfg["H"], cfg["W"], cfg["focal"], cfg["c2w"])
    points, t_vals, deltas = sample_points(
        rays_o, rays_d, cfg["near"], cfg["far"], cfg["num_samples"]
    )
    sigmas, rgbs = query_radiance_field(points)
    pred_rgb, pred_depth, weights = volume_render(sigmas, rgbs, deltas, t_vals)

    opacity = np.sum(weights, axis=-1)
    metrics = {
        "rgb_min": float(pred_rgb.min()),
        "rgb_max": float(pred_rgb.max()),
        "depth_min": float(pred_depth.min()),
        "depth_max": float(pred_depth.max()),
        "mean_opacity": float(opacity.mean()),
    }

    np.save(os.path.join(results_dir, "mini_nerf_rgb.npy"), pred_rgb)
    np.save(os.path.join(results_dir, "mini_nerf_depth.npy"), pred_depth)
    np.save(os.path.join(results_dir, "mini_nerf_metrics.npy"), metrics)

    return {
        "cfg": cfg,
        "rays_o": rays_o,
        "rays_d": rays_d,
        "points": points,
        "t_vals": t_vals,
        "deltas": deltas,
        "pred_rgb": pred_rgb,
        "pred_depth": pred_depth,
        "weights": weights,
        "metrics": metrics,
    }


if __name__ == "__main__":
    outputs = run()
    print("pred_rgb shape:", outputs["pred_rgb"].shape)
    print("pred_depth shape:", outputs["pred_depth"].shape)
    print("metrics:", outputs["metrics"])
