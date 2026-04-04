import numpy as np
from utils import draw_save_plane_with_points, normalize
import math

if __name__ == "__main__":


    np.random.seed(0)
    # load data, total 130 points inlcuding 100 inliers and 30 outliers
    # to simplify this problem, we provide the number of inliers and outliers here

    noise_points = np.loadtxt("HM1_ransac_points.txt")


    #RANSAC
    # Please formulate the palnace function as:  A*x+B*y+C*z+D=0     
    inliers, outliers = 100, 30
    total = inliers + outliers
    sample_time = math.ceil(math.log(1 - 0.999) / math.log(1 - (inliers / total) ** 3))
    sample_time = int(sample_time)
    # the minimal time that can guarantee the probability of at least one hypothesis does not contain any outliers is larger than 99.9%
    distance_threshold = 0.05
    # sample points group
    N = noise_points.shape[0]
    rng = np.random.default_rng()
    keys = rng.random((sample_time, N))
    sampled_idx = np.argsort(keys, axis=1)[:, :3]
    # estimate the plane with sampled points group
    sampled_pts = noise_points[sampled_idx]
    p1 = sampled_pts[:, 0, :]
    p2 = sampled_pts[:, 1, :]
    p3 = sampled_pts[:, 2, :]
    v1 = p2 - p1
    v2 = p3 - p1
    normals = np.cross(v1, v2)
    norms = np.sqrt(np.sum(normals**2, axis=1))
    valid = norms > 1e-8
    norms[~valid] = 1.0
    D = -np.sum(normals * p1, axis=1)
    #evaluate inliers (with point-to-plance distance < distance_threshold)
    numer = normals @ noise_points.T + D[:, None]
    dists = np.abs(numer) / norms[:, None]
    dists[~valid, :] = np.inf
    inlier_mask = (dists <= distance_threshold)
    inlier_counts = inlier_mask.sum(axis=1)
    # minimize the sum of squared perpendicular distances of all inliers with least-squared method 
    best_idx = int(np.argmax(inlier_counts))
    best_inliers = noise_points[inlier_mask[best_idx]]
    centroid = best_inliers.mean(axis=0)
    centered_inliers = best_inliers - centroid
    _, _, vh = np.linalg.svd(centered_inliers)
    normal_vec = vh[-1, :]
    D = -normal_vec @ centroid
    pf = np.append(normal_vec, D)
    # draw the estimated plane with points and save the results 
    # check the utils.py for more details
    # pf: [A,B,C,D] contains the parameters of palnace function  A*x+B*y+C*z+D=0  
    pf = normalize(pf)
    draw_save_plane_with_points(pf, noise_points,"result/HM1_RANSAC_fig.png") 
    np.savetxt("result/HM1_RANSAC_plane.txt", pf)
    np.savetxt('result/HM1_RANSAC_sample_time.txt', np.array([sample_time]))
