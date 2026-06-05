import os

import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment


def uniform_sampling_from_mesh(vertices, faces, sample_num, rng=None):
    rng = np.random.default_rng() if rng is None else rng

    triangles = vertices[faces]
    cross_products = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = 0.5 * np.linalg.norm(cross_products, axis=1)
    prob = area / area.sum()

    sampled_face_ids = rng.choice(faces.shape[0], size=sample_num, p=prob)
    sampled_triangles = triangles[sampled_face_ids]

    r1 = np.sqrt(rng.random((sample_num, 1)))
    r2 = rng.random((sample_num, 1))
    uniform_pc = (
        (1.0 - r1) * sampled_triangles[:, 0]
        + r1 * (1.0 - r2) * sampled_triangles[:, 1]
        + r1 * r2 * sampled_triangles[:, 2]
    )
    return area, prob, uniform_pc


def farthest_point_sampling(pc, sample_num):
    if sample_num > pc.shape[0]:
        raise ValueError("sample_num must not exceed the number of input points.")

    selected = np.zeros(sample_num, dtype=np.int32)
    centroid = pc.mean(axis=0)
    selected[0] = np.argmax(np.linalg.norm(pc - centroid, axis=1))

    min_dist = np.linalg.norm(pc - pc[selected[0]], axis=1)
    for i in range(1, sample_num):
        selected[i] = np.argmax(min_dist)
        new_dist = np.linalg.norm(pc - pc[selected[i]], axis=1)
        min_dist = np.minimum(min_dist, new_dist)

    return pc[selected]


def chamfer_distance(pc1, pc2):
    pairwise_dist = np.linalg.norm(pc1[:, None, :] - pc2[None, :, :], axis=-1)
    forward = pairwise_dist.min(axis=1).mean()
    backward = pairwise_dist.min(axis=0).mean()
    return float(0.5 * (forward + backward))


def earth_movers_distance(pc1, pc2, solver=None):
    if solver is not None:
        return float(solver(pc1, pc2))

    pairwise_dist = np.linalg.norm(pc1[:, None, :] - pc2[None, :, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(pairwise_dist)
    return float(pairwise_dist[row_ind, col_ind].mean())


def compute_sampling_metrics(
    vertices,
    faces,
    repeats=5,
    uniform_num=512,
    init_sample_num=2000,
    fps_num=512,
    seed=0,
    emd_solver=None,
):
    rng = np.random.default_rng(seed)
    chamfer_values = []
    emd_values = []

    for _ in range(repeats):
        _, _, uniform_pc = uniform_sampling_from_mesh(vertices, faces, uniform_num, rng)
        _, _, dense_pc = uniform_sampling_from_mesh(vertices, faces, init_sample_num, rng)
        fps_pc = farthest_point_sampling(dense_pc, fps_num)

        chamfer_values.append(chamfer_distance(uniform_pc, fps_pc))
        emd_values.append(earth_movers_distance(uniform_pc, fps_pc, solver=emd_solver))

    chamfer_values = np.asarray(chamfer_values, dtype=np.float64)
    emd_values = np.asarray(emd_values, dtype=np.float64)
    return {
        "CD_mean": float(chamfer_values.mean()),
        "CD_var": float(chamfer_values.var()),
        "EMD_mean": float(emd_values.mean()),
        "EMD_var": float(emd_values.var()),
    }


def run(base_dir=None, results_dir=None, seed=0):
    base_dir = os.path.dirname(os.path.abspath(__file__)) if base_dir is None else base_dir
    results_dir = (
        os.path.abspath(os.path.join(base_dir, "..", "results"))
        if results_dir is None
        else results_dir
    )
    os.makedirs(results_dir, exist_ok=True)

    mesh = trimesh.load(os.path.join(base_dir, "spot.obj"))
    rng = np.random.default_rng(seed)

    area, prob, uniform_pc = uniform_sampling_from_mesh(mesh.vertices, mesh.faces, 512, rng)
    np.savetxt(os.path.join(base_dir, "uniform_sampling_vis.txt"), uniform_pc)
    np.save(
        os.path.join(results_dir, "uniform_sampling_results.npy"),
        {"area": area, "prob": prob, "pc": uniform_pc},
    )

    _, _, dense_pc = uniform_sampling_from_mesh(mesh.vertices, mesh.faces, 2000, rng)
    fps_pc = farthest_point_sampling(dense_pc, 512)
    np.savetxt(os.path.join(base_dir, "fps_vis.txt"), fps_pc)
    np.save(os.path.join(results_dir, "fps_results.npy"), fps_pc)

    try:
        from earthmover.earthmover import earthmover_distance as emd_solver
    except ImportError:
        emd_solver = None

    metrics = compute_sampling_metrics(
        mesh.vertices,
        mesh.faces,
        repeats=5,
        uniform_num=512,
        init_sample_num=2000,
        fps_num=512,
        seed=seed,
        emd_solver=emd_solver,
    )
    np.save(os.path.join(results_dir, "metrics.npy"), metrics)

    return {
        "mesh": mesh,
        "area": area,
        "prob": prob,
        "uniform_pc": uniform_pc,
        "fps_pc": fps_pc,
        "metrics": metrics,
    }


if __name__ == "__main__":
    outputs = run()
    print("faces shape:", outputs["mesh"].faces.shape)
    print("area shape:", outputs["area"].shape)
    print("prob shape:", outputs["prob"].shape)
    print("uniform pc shape:", outputs["uniform_pc"].shape)
    print("fps pc shape:", outputs["fps_pc"].shape)
    print("metrics:", outputs["metrics"])
