import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

from transforms3d.quaternions import qmult, mat2quat
from transforms3d.euler import quat2euler, mat2euler

torch.set_default_dtype(torch.float64)

# Read pickle data file
def read_data(fname):
    with open(fname, "rb") as f:
        return pickle.load(f, encoding="latin1")

# Calibrate IMU and remove biases
def calibration(imud, time, static_sec=6.0):
    raw_acc = imud[1:4]
    raw_gyro = imud[4:7]

    vref = 3300.0
    acc_sens = 330.0
    gyro_sens = 3.33 * 180 / np.pi

    scalef_acc = vref / (1023 * acc_sens)
    scalef_gyro = vref / (1023 * gyro_sens)

    static_mask = time < static_sec

    bias_gyro_raw = raw_gyro[:, static_mask].mean(axis=1)
    bias_acc_raw = raw_acc[:, static_mask].mean(axis=1)

    scaled_acc = raw_acc * scalef_acc
    bias_acc = bias_acc_raw * scalef_acc - np.array([0.0, 0.0, 1.0])

    value_acc = scaled_acc - bias_acc[:, None]
    value_gyro = (raw_gyro - bias_gyro_raw[:, None]) * scalef_gyro

    return value_acc, value_gyro

# Skew-symmetric matrix for cross product
def hat(v):
    vx, vy, vz = v
    return np.array([[0.0, -vz, vy],
                     [vz, 0.0, -vx],
                     [-vy, vx, 0.0]])

# Rodrigues rotation from axis-angle
def rotation_matrix(axis):
    theta = np.linalg.norm(axis)
    if theta < 1e-12:
        return np.eye(3)
    K = hat(axis / theta)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

# Integrate gyroscope to get quaternion trajectory
def integrate_gyro(time, value_gyro):
    N = value_gyro.shape[1]
    q = np.zeros((N, 4))
    q[0] = np.array([1.0, 0.0, 0.0, 0.0])

    for i in range(N - 1):
        dt = time[i+1] - time[i]
        axis = value_gyro[:, i] * dt
        R = rotation_matrix(axis)
        dq = mat2quat(R)

        q[i+1] = qmult(q[i], dq)
        q[i+1] /= np.linalg.norm(q[i+1]) + 1e-12
        if np.dot(q[i+1], q[i]) < 0:
            q[i+1] = -q[i+1]
    return q

# Quaternion multiplication (torch)
def qmul(a, b):
    w1, x1, y1, z1 = a[...,0], a[...,1], a[...,2], a[...,3]
    w2, x2, y2, z2 = b[...,0], b[...,1], b[...,2], b[...,3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w,x,y,z], dim=-1)

# Quaternion conjugate
def qconj(q):
    return torch.stack([q[...,0], -q[...,1], -q[...,2], -q[...,3]], dim=-1)

# Normalize quaternion
def qnorm(q):
    return q / (torch.linalg.norm(q, dim=-1, keepdim=True) + 1e-12)

# Quaternion exponential map
def qexp(v):
    theta = torch.linalg.norm(v, dim=-1, keepdim=True)
    small = (theta < 1e-12).squeeze(-1)
    axis = v / (theta + 1e-12)

    w = torch.cos(theta)
    xyz = axis * torch.sin(theta)
    q = torch.cat([w, xyz], dim=-1)

    if small.any():
        q_small = torch.cat([torch.ones_like(theta), v], dim=-1)
        q = torch.where(small.unsqueeze(-1), q_small, q)
    return q

# Quaternion logarithm map
def qlog(q):
    w = torch.clamp(q[..., 0:1], -1.0, 1.0)
    v = q[..., 1:]
    vnorm = torch.linalg.norm(v, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vnorm, w.abs() + 1e-12)
    axis = v / (vnorm + 1e-12)
    r = axis * angle
    r = torch.where(vnorm < 1e-12, torch.zeros_like(r), r)
    return r

# Motion model for quaternion
def f_motion(q_t, dt, w_t):
    dq = qexp(0.5 * dt * w_t)
    return qmul(q_t, dq)

# Predict gravity direction in body frame
def h_gravity(q_t):
    g = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=q_t.dtype, device=q_t.device)
    g = g.expand(q_t.shape[0], 4)
    return qmul(qmul(qconj(q_t), g), q_t)

# Cost function for trajectory optimization
def cost(q, time, omega, acc, w_motion=1.0, w_acc=1.0):
    T = q.shape[0]
    dt = (time[1:] - time[:-1]).unsqueeze(-1)

    q_t = q[:-1]
    q_next = q[1:]
    w_t = omega[:-1]

    q_pred = f_motion(q_t, dt, w_t)
    q_rel = qmul(qconj(q_next), q_pred)
    q_rel = qnorm(q_rel)

    r = 2.0 * qlog(q_rel)
    motion_term = 0.5 * torch.sum(r * r)

    a_meas = torch.cat([torch.zeros((T,1), dtype=q.dtype, device=q.device), acc], dim=-1)
    a_pred = h_gravity(q)
    e = a_meas - a_pred
    acc_term = 0.5 * torch.sum(e * e)

    total = w_motion * motion_term + w_acc * acc_term
    return total, motion_term, acc_term

# Projected gradient descent on quaternions
def optimize(q_init, time, omega, acc, iters=200, lr=1e-2, w_motion=1.0, w_acc=1.0, print_every=25):
    time_t = torch.tensor(time)
    omega_t = torch.tensor(omega.T)
    acc_t = torch.tensor(acc.T)
    q = torch.tensor(q_init, requires_grad=True)

    for it in range(iters):
        total, motion_c, acc_c = cost(q, time_t, omega_t, acc_t, w_motion=w_motion, w_acc=w_acc)

        if q.grad is not None:
            q.grad.zero_()
        total.backward()

        with torch.no_grad():
            q -= lr * q.grad
            q[:] = qnorm(q)

            dots = torch.sum(q[1:] * q[:-1], dim=-1)
            flip = dots < 0
            q[1:][flip] = -q[1:][flip]

        if it % print_every == 0 or it == iters - 1:
            print(f"iter {it:4d} | total={total.item():.4f} motion={motion_c.item():.4f} acc={acc_c.item():.4f}")

    return q.detach().numpy()

# Plot training RPY vs VICON
def plot_train_rpy_vs_vicon(q_traj, vicd, time, dataset, outdir="results"):
    os.makedirs(outdir, exist_ok=True)

    N = q_traj.shape[0]
    rpy_est = np.zeros((N, 3))
    for i in range(N):
        rpy_est[i] = quat2euler(q_traj[i], axes="sxyz")

    vicon_time = vicd["ts"].reshape(-1)
    vicon_time = vicon_time - vicon_time[0]
    R = vicd["rots"]
    M = R.shape[2]
    rpy_v = np.zeros((M, 3))
    for i in range(M):
        rpy_v[i] = mat2euler(R[:,:,i], axes="sxyz")

    idx = np.searchsorted(vicon_time, time, side="left")
    idx = np.clip(idx, 0, M-1)
    rpy_gt = rpy_v[idx]

    for k in range(3):
        rpy_est[:,k] = np.unwrap(rpy_est[:,k])
        rpy_gt[:,k]  = np.unwrap(rpy_gt[:,k])

    labels = ["Roll", "Pitch", "Yaw"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    for k in range(3):
        axes[k].plot(time, rpy_gt[:,k], "b-", label="VICON (GT)", linewidth=2)
        axes[k].plot(time, rpy_est[:,k], "r--", label="Estimate", linewidth=1.5)
        axes[k].set_ylabel(f"{labels[k]} (rad)")
        axes[k].grid(True)
        axes[k].legend(loc="best")

    axes[2].set_xlabel("Time (s)")
    fig.suptitle(f"Orientation Estimate vs VICON — Training Dataset {dataset}", fontsize=14)

    fname = os.path.join(outdir, f"train_rpy_dataset_{dataset}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")

# Plot test RPY only
def plot_test_rpy(q_traj, time, dataset, outdir="results"):
    os.makedirs(outdir, exist_ok=True)

    N = q_traj.shape[0]
    rpy = np.zeros((N, 3))
    for i in range(N):
        rpy[i] = quat2euler(q_traj[i], axes="sxyz")
    for k in range(3):
        rpy[:,k] = np.unwrap(rpy[:,k])

    labels = ["Roll", "Pitch", "Yaw"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for k in range(3):
        axes[k].plot(time, rpy[:,k], "r--", linewidth=1.5, label="Estimate")
        axes[k].set_ylabel(f"{labels[k]} (rad)")
        axes[k].grid(True)
        axes[k].legend(loc="best")
    axes[2].set_xlabel("Time (s)")
    fig.suptitle(f"Orientation Estimate — Test Dataset {dataset}", fontsize=14)

    fname = os.path.join(outdir, f"test_rpy_dataset_{dataset}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")

# Convert quaternion to rotation matrix (NumPy)
def quat_to_R_np(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ], dtype=np.float64)

# Build equirectangular panorama
def build_panorama(camd, time_imu, q_traj, pano_H=240, pano_W=3000, fov_deg=70.0):
    imgs = camd["cam"]
    ts_cam = camd["ts"].reshape(-1)
    ts_cam = ts_cam - ts_cam[0]

    H, W, _, N = imgs.shape

    fov = np.deg2rad(fov_deg)
    fx = (W/2.0) / np.tan(fov/2.0)
    fy = fx
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    us = np.arange(W)
    vs = np.arange(H)
    uu, vv = np.meshgrid(us, vs)

    x = (uu - cx) / fx
    y = (vv - cy) / fy
    z = np.ones_like(x)

    rays_cam = np.stack([x, y, z], axis=-1)
    rays_cam /= np.linalg.norm(rays_cam, axis=-1, keepdims=True) + 1e-12

    pano = np.zeros((pano_H, pano_W, 3), dtype=np.uint8)

    def dir_to_uv(d):
        dx, dy, dz = d[...,0], d[...,1], d[...,2]
        yaw = np.arctan2(dy, dx)
        pitch = np.arcsin(np.clip(dz, -1, 1))
        u = (yaw + np.pi) / (2*np.pi) * (pano_W - 1)
        v = (np.pi/2 - pitch) / np.pi * (pano_H - 1)
        return u.astype(np.int32), v.astype(np.int32)

    time_imu = np.asarray(time_imu).reshape(-1)

    for k in range(N):
        t_cam = ts_cam[k]
        idx = np.searchsorted(time_imu, t_cam, side="right") - 1
        idx = int(np.clip(idx, 0, len(time_imu)-1))

        R = quat_to_R_np(q_traj[idx])
        img = imgs[:,:,:,k]

        rays_w = rays_cam @ R.T
        u, v = dir_to_uv(rays_w)
        u = np.clip(u, 0, pano_W-1)
        v = np.clip(v, 0, pano_H-1)

        pano[v, u] = img

    return pano

# Save panorama image
def save_panorama(pano, title, fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.figure(figsize=(14, 4))
    plt.imshow(pano)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

# Main pipeline
def main():
    outdir = "results"
    os.makedirs(outdir, exist_ok=True)

    train_rpy_datasets = [str(i) for i in range(1, 10)]
    print("\n=== GENERATING ALL TRAINING RPY PLOTS (1–9) ===\n")

    for d in train_rpy_datasets:
        print(f"\n--- Dataset {d}: orientation estimation ---")

        imud = read_data(f"../data/trainset/imu/imuRaw{d}.p")
        vicd = read_data(f"../data/trainset/vicon/viconRot{d}.p")

        time = imud[0] - imud[0][0]
        value_acc, value_gyro = calibration(imud, time)

        q0 = integrate_gyro(time, value_gyro)
        q_opt = optimize(q0, time, value_gyro, value_acc,
                         iters=200, lr=1e-2,
                         w_motion=1.0, w_acc=1.0)

        plot_train_rpy_vs_vicon(q_opt, vicd, time, d, outdir=outdir)

    train_cam_datasets = ["1", "2", "8", "9"]
    print("\n=== GENERATING ALL TRAINING PANORAMAS (1,2,8,9) ===\n")

    for d in train_cam_datasets:
        print(f"\n--- Dataset {d}: panorama ---")

        camd = read_data(f"../data/trainset/cam/cam{d}.p")
        imud = read_data(f"../data/trainset/imu/imuRaw{d}.p")

        time = imud[0] - imud[0][0]
        value_acc, value_gyro = calibration(imud, time)

        q0 = integrate_gyro(time, value_gyro)
        q_opt = optimize(q0, time, value_gyro, value_acc,
                         iters=200, lr=1e-2)

        pano = build_panorama(camd, time, q_opt,
                              pano_H=240, pano_W=3000, fov_deg=70.0)

        save_panorama(
            pano,
            title=f"Panorama — Training Dataset {d}",
            fname=os.path.join(outdir, f"train_panorama_dataset_{d}.png")
        )

    test_rpy_datasets = ["10", "11"]
    test_cam_datasets = ["10", "11"]

    print("\n=== GENERATING TEST RESULTS ===\n")

    for d in test_rpy_datasets:
        print(f"\n--- Test Dataset {d}: orientation ---")
        imud = read_data(f"../data/testset/imu/imuRaw{d}.p")
        time = imud[0] - imud[0][0]
        value_acc, value_gyro = calibration(imud, time)

        q0 = integrate_gyro(time, value_gyro)
        q_opt = optimize(q0, time, value_gyro, value_acc, iters=200, lr=1e-2)

        plot_test_rpy(q_opt, time, d, outdir=outdir)

    for d in test_cam_datasets:
        print(f"\n--- Test Dataset {d}: panorama ---")
        camd = read_data(f"../data/testset/cam/cam{d}.p")
        imud = read_data(f"../data/testset/imu/imuRaw{d}.p")

        time = imud[0] - imud[0][0]
        value_acc, value_gyro = calibration(imud, time)
        q0 = integrate_gyro(time, value_gyro)
        q_opt = optimize(q0, time, value_gyro, value_acc, iters=200, lr=1e-2)

        pano = build_panorama(camd, time, q_opt, pano_H=240, pano_W=3000, fov_deg=70.0)

        save_panorama(
            pano,
            title=f"Panorama — Test Dataset {d}",
            fname=os.path.join(outdir, f"test_panorama_dataset_{d}.png")
        )

if __name__ == "__main__":
    main()

