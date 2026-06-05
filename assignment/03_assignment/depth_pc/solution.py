import os

import cv2
import numpy as np


DEPTH_SCALE = 0.00012498664727900177


def depth2pc(depth, seg, K):
    mask = (seg > 0) & (depth > 0)
    v, u = np.nonzero(mask)
    z = depth[mask]

    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1)


def random_sample(pc, num, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    num = min(num, pc.shape[0])
    permu = rng.permutation(pc.shape[0])
    return pc[permu][:num]


def one_way_chamfer_distance(source_pc, target_pc):
    pairwise_dist = np.linalg.norm(
        source_pc[:, None, :] - target_pc[None, :, :], axis=-1
    )
    return float(pairwise_dist.min(axis=1).mean())


def load_inputs(base_dir=None):
    base_dir = os.path.dirname(os.path.abspath(__file__)) if base_dir is None else base_dir

    depth_img = cv2.imread(os.path.join(base_dir, "depth.png"))
    depth = depth_img[:, :, 2].astype(np.int32) + depth_img[:, :, 1].astype(np.int32) * 256
    depth = depth * DEPTH_SCALE

    seg = cv2.imread(os.path.join(base_dir, "seg.png"))[..., 0]
    K = np.load(os.path.join(base_dir, "intrinsic.npy"))
    full_pc = np.loadtxt(os.path.join(base_dir, "aligned_full_pc.txt"))
    return depth, seg, K, full_pc


def run(base_dir=None, results_dir=None, sample_num=2048, seed=0):
    base_dir = os.path.dirname(os.path.abspath(__file__)) if base_dir is None else base_dir
    results_dir = (
        os.path.abspath(os.path.join(base_dir, "..", "results"))
        if results_dir is None
        else results_dir
    )
    os.makedirs(results_dir, exist_ok=True)

    depth, seg, K, full_pc = load_inputs(base_dir)
    partial_pc = depth2pc(depth, seg, K)
    np.savetxt(os.path.join(results_dir, "pc_from_depth.txt"), partial_pc)

    rng = np.random.default_rng(seed)
    partial_pc_sampled = random_sample(partial_pc, sample_num, rng)
    full_pc_sampled = random_sample(full_pc, sample_num, rng)
    one_way_cd = one_way_chamfer_distance(partial_pc_sampled, full_pc_sampled)
    np.savetxt(os.path.join(results_dir, "one_way_CD.txt"), np.array([one_way_cd]))

    return {
        "partial_pc": partial_pc,
        "partial_pc_sampled": partial_pc_sampled,
        "full_pc_sampled": full_pc_sampled,
        "one_way_CD": one_way_cd,
        "K": K,
    }


if __name__ == "__main__":
    outputs = run()
    print("partial_pc shape:", outputs["partial_pc"].shape)
    print("one way chamfer distance:", outputs["one_way_CD"])
