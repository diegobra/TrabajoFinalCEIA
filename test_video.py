# test_video.py
# Genera un video cinematográfico (sin modificar emisores)
# Recibe los mismos parámetros que test_interactive.py (Hydra, cfg.test_rendering, etc.)
# Movimiento: (1) avance paramétrico; (2) pan izquierda; (3) pan derecha (paramétricos)
import logging
import os
from pathlib import Path
import gc
import numpy as np
import cv2
from tqdm import tqdm

import drjit as dr
import mitsuba as mi
import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from common import (
    configure_compute,
    create_integrator,
    create_loss_function,
    load_dataset,
    prepare_learned_objects,
)
from nerad.integrator.highquality import HighQuality
from nerad.model.config import ObjectConfig, TestConfig, TrainConfig
from nerad.utils.render_utils import render_and_save_image
from nerad.utils.sensor_utils import create_sensor

logger = logging.getLogger(__name__)

# =========================
# Utilidades de cámara
# =========================
def look_at(eye, target, up=np.array([0, 1, 0], dtype=np.float32)):
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up); r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    M = np.eye(4, dtype=np.float32)
    M[0, :3] = r; M[1, :3] = u; M[2, :3] = -f
    M[:3, 3] = eye
    return M

def lerp(a, b, t): return a * (1.0 - t) + b * t
def smoothstep01(t): return t * t * (3 - 2 * t)
def smootherstep01(t):  # más suave (C2)
    return t * t * t * (t * (t * 6 - 15) + 10)

def decompose_to_world(T):
    T = np.array(T, dtype=np.float32)
    return T[:3, :3], T[:3, 3]

def compose_to_world(R, t):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R; T[:3, 3] = t
    return T

def apply_local_delta(base_T, delta_pos=(0,0,0), yaw_deg=0.0, pitch_deg=0.0):
    """Movimiento local de cámara (adelante = -Z local). dz>0 avanza.
       yaw/pitch en grados sobre el sistema local."""
    base_T = np.array(base_T, dtype=np.float32)
    R0, t0 = decompose_to_world(base_T)
    yaw = np.deg2rad(yaw_deg); pitch = np.deg2rad(pitch_deg)
    Ry = np.array([[ np.cos(yaw), 0, np.sin(yaw)],
                   [ 0,           1, 0          ],
                   [-np.sin(yaw), 0, np.cos(yaw)]], dtype=np.float32)
    Rx = np.array([[1, 0,            0           ],
                   [0, np.cos(pitch),-np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]], dtype=np.float32)
    R = R0 @ Ry @ Rx

    right, up, fwd_neg = R[:,0], R[:,1], R[:,2]
    dx, dy, dz = delta_pos
    t = t0 + dx*right + dy*up + dz*(-fwd_neg)  # dz>0 avanza hacia adelante

    T = compose_to_world(R, t)
    target = t + (-fwd_neg) * 1.0
    return T, target

# =========================
# Conversión de imagen robusta
# =========================
def tensor_to_frame(outputs):
    """Convierte outputs a BGR u8 con fallback si un buffer viene negro.
    Intenta outputs[1] sRGB; si no, lineal + Reinhard; si no, outputs[0]."""
    def as_bgr_u8_from_idx(idx, gamma=True):
        img_mi = mi.Bitmap(outputs[idx].numpy())
        img_mi = img_mi.convert(mi.Bitmap.PixelFormat.RGB,
                                mi.Struct.Type.UInt8 if gamma else mi.Struct.Type.Float32,
                                gamma)
        arr = np.array(img_mi)
        if gamma:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        # Reinhard
        arr = arr / (1.0 + arr); arr = np.clip(arr, 0, 1)
        arr = (arr*255).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    try:
        bgr = as_bgr_u8_from_idx(1, gamma=True)
        if bgr.mean() > 0.5: return bgr
    except: pass
    try:
        bgr = as_bgr_u8_from_idx(1, gamma=False)
        if bgr.mean() > 0.5: return bgr
    except: pass
    try:
        bgr = as_bgr_u8_from_idx(0, gamma=True)
        if bgr.mean() > 0.5: return bgr
    except: pass
    bgr = as_bgr_u8_from_idx(0, gamma=False)
    return bgr

# =========================
# Tomas (paramétricas, smooth)
# =========================
def shot1_advance_param(base_T, t, dist_m):
    """Avance hacia el interior: dist_m en metros (param)."""
    k = smootherstep01(t)
    T, target = apply_local_delta(base_T, delta_pos=(0, 0, -dist_m * k), yaw_deg=0, pitch_deg=0)
    fov = lerp(35.0, 30.0, k)
    focus = 1.2
    aperture = 0.02
    return T, target, fov, focus, aperture

def shot_pan_between(base_T, t, yaw_start_deg, yaw_end_deg, pitch_deg=0.0):
    """Panorámica entre dos yaw relativos (grados) sobre base_T."""
    k = smootherstep01(t)
    yaw = lerp(yaw_start_deg, yaw_end_deg, k)
    T, target = apply_local_delta(base_T, delta_pos=(0.0, 0.0, 0.0), yaw_deg=yaw, pitch_deg=pitch_deg)
    fov = 30.0
    focus = 1.5
    aperture = 0.02
    return T, target, fov, focus, aperture

def shot_return_center_look_up(base_T, t, start_yaw_deg, look_up_deg=12.0, pedestal=0.08):
    """Vuelve suavemente a yaw=0 desde start_yaw_deg y eleva la mirada para mostrar la fuente de luz."""
    k = smootherstep01(t)
    yaw = lerp(start_yaw_deg, 0.0, k)  # vuelve al centro
    # leve pedestal hacia arriba mientras mira al techo
    T, target = apply_local_delta(
        base_T,
        delta_pos=(0.0, pedestal * k, 0.0),
        yaw_deg=yaw,
        pitch_deg=lerp(0.0, -abs(look_up_deg), k)
    )
    fov = lerp(30.0, 28.0, k)
    focus = 1.5
    aperture = 0.02
    return T, target, fov, focus, aperture


# =========================
# Render a PNGs (+ MP4 opcional)
# =========================
def cinematic_render_to_video(cfg_test_rendering, scene, transforms, images, test_integrators, out_root,
                              fps=24, width_override=None, use_videowriter=False, fourcc_str="mp4v"):
    """(1) Avanza (paramétrico) → (2) pan izquierda → (3) pan derecha.
       Exporta PNGs y opcionalmente MP4. No toca emisores."""
    test_integrators['image'].set_custom_config(cfg_test_rendering, interactive_test=True)

    # --- Suavidad temporal: pasos más pequeños ---
    fps = int(fps)
    temporal_subdiv = int(getattr(cfg_test_rendering, 'temporal_subdiv', 1))  # 3-4 recomendado
    if temporal_subdiv < 1: temporal_subdiv = 1
    effective_fps = fps * temporal_subdiv

    # --- Duraciones por toma (seg) ---
    shot_secs = [12.0, 8.0, 8.0]  # avance / pan izq / pan der (suaves)

    # --- Resolución ---
    width_target  = int(width_override) if width_override else int(cfg_test_rendering.width)
    height_target =  576 #int(getattr(cfg_test_rendering, 'width', int(width_target * 9 / 16)))

    frames_dir = out_root / "video_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    out_video_path = out_root / "living_room_cinematic.mp4"
    if use_videowriter:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(out_video_path), fourcc, effective_fps, (width_target, height_target))
        if not writer.isOpened():
            print("⚠️  VideoWriter no se pudo abrir. Sigo con PNGs + ffmpeg.")
            writer = None

    test_integrator = test_integrators["image"]

    # --- Parámetros del usuario (Hydra) ---
    move_dist_m  = float(getattr(cfg_test_rendering, 'move_dist_m', 2.2))     # metros de avance
    pan_left_deg = float(getattr(cfg_test_rendering, 'pan_left_deg', 20.0))   # giro a izquierda
    pan_right_deg= float(getattr(cfg_test_rendering, 'pan_right_deg', 25.0))  # giro a derecha
    center_look_up_deg = float(getattr(cfg_test_rendering, 'center_look_up_deg', 12.0))

    # --- Base y punto de llegada del avance ---
    base_view_key = '0' if '0' in transforms else list(transforms.keys())[0]
    base_T = np.array(transforms[base_view_key]['to_world'], dtype=np.float32)
    base_T_fwd_end = shot1_advance_param(base_T, 1.0, move_dist_m)[0]

    # --- Secuencia de tomas ---
    shots = [
        ("advance",   lambda t: shot1_advance_param(base_T,        t, move_dist_m)),
        ("pan_left",  lambda t: shot_pan_between   (base_T_fwd_end, t,  0.0, -abs(pan_left_deg),  0.0)),
        ("pan_right", lambda t: shot_pan_between   (base_T_fwd_end, t, -abs(pan_left_deg),  abs(pan_right_deg), 0.0)),
    ]

        # Añadir toma final: volver al centro y mirar hacia arriba.
    # Usamos base_T_fwd_end (punto tras el avance) y partimos desde el yaw final del paneo a la derecha.
    shots.append((
        "center_up",
        lambda t: shot_return_center_look_up(base_T_fwd_end, t, pan_right_deg, center_look_up_deg)
    ))

    # Y extendé las duraciones:
    shot_secs.append(6.0)  # p.ej. 6 segundos para el remate

    # --- Render ---
    frame_idx = 0
    for si, (name, fn) in enumerate(shots):
        nf = int(shot_secs[si] * fps * temporal_subdiv)
        for i in tqdm(range(nf), desc=f"Rendering {name}"):
            t = i / max(1, (nf - 1))
            T, target, fov, focus, aperture = fn(t)

            tr = {
                "to_world": T.tolist(),
                "fov": float(fov),
                "aperture_radius": float(aperture),
                "focus_distance": float(focus),
                "shutter_open": 0.0,
                "shutter_open_time": 0.0,
            }

            dr.flush_malloc_cache(); torch.cuda.empty_cache(); gc.collect()

            sensor = create_sensor(cfg_test_rendering.width, tr)
            outputs = render_and_save_image(
                out_root / "video_tmp",
                f"frame_{frame_idx:05d}",
                scene,
                test_integrator,
                cfg_test_rendering,
                sensor,
                save_image_to_disk=False,
            )

            frame_bgr = tensor_to_frame(outputs)

            png_path = frames_dir / f"frame_{frame_idx:05d}.png"
            ok = cv2.imwrite(str(png_path), frame_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if not ok:
                raise RuntimeError(f"No se pudo escribir el frame {png_path}")

            if writer is not None:
                if frame_bgr.shape[:2] != (height_target, width_target):
                    frame_bgr = cv2.resize(frame_bgr, (width_target, height_target), interpolation=cv2.INTER_AREA)
                writer.write(frame_bgr)

            frame_idx += 1

    if writer is not None:
        writer.release()

    print("\n✅ Frames exportados en:", str(frames_dir))
    print("Para generar el MP4 manteniendo la **misma duración** (más suave):")
    print(f"ffmpeg -y -framerate {effective_fps} -i '{frames_dir}/frame_%05d.png' -vf 'scale={width_target}:-2' -c:v libx264 -pix_fmt yuv420p -b:v 12M '{out_video_path}'")
    print("\nSi preferís que el video quede **más largo** a 24 fps reales:")
    print(f"ffmpeg -y -framerate {fps} -i '{frames_dir}/frame_%05d.png' -vf 'scale={width_target}:-2' -c:v libx264 -pix_fmt yuv420p -b:v 12M '{out_video_path}'")

# =========================
# Hydra main
# =========================
@hydra.main(version_base="1.2", config_path="config", config_name="test")
def main(cfg: TestConfig = None):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    use_hq = True

    device = os.environ.get("TORCH_DEVICE", "cuda:0")
    out_root = Path(HydraConfig.get().runtime.output_dir)  # experiment/test/ckpt

    # merge config from training
    train_root = out_root.parent.parent
    train_cfg = OmegaConf.load(train_root / ".hydra/config.yaml")
    cfg = merge_config(cfg, train_cfg)

    configure_compute(cfg.compute)

    scene, transforms, images, learned_modules = load_dataset(cfg.dataset, device)

    test_rendering = cfg.test_rendering
    test_integrators = {}
    for key, rendering in test_rendering.items():
        is_nerad = "nerad" in rendering.integrator
        integrator_injection = {}
        if is_nerad:
            residual_loss_function = create_loss_function(ObjectConfig("l2", {}), 0)  # dummy
            integrator_injection["residual_function"] = residual_loss_function
        integrator_function_injection = {"device": device}

        integrator = create_integrator(
            rendering, scene, cfg.dataset.scene,
            post_init_injection=integrator_injection,
            kwargs_injection=integrator_function_injection,
        )
        test_integrators[key] = integrator

    _ = prepare_learned_objects(
        scene, test_integrators["image"], learned_modules, None,
        train_root / "checkpoints" / f"{cfg.ckpt}.ckpt", device,
    )

    if use_hq:
        block_size = cfg.blocksize
        logger.info(f"High quality renderer being used at block size: {block_size}")
        test_integrators = {k: HighQuality(block_size, v) for k, v in test_integrators.items()}

    cfg_test_rendering = cfg.test_rendering["image"]

    # Render
    cinematic_render_to_video(
        cfg_test_rendering, scene, transforms, images, test_integrators, out_root,
        fps=24,
        width_override=None,   # o 1920 para 1080p
        use_videowriter=False, # True si tu códec funciona bien
        fourcc_str="mp4v",
    )

def merge_config(test: TestConfig, train: TrainConfig) -> TestConfig:
    test.dataset = OmegaConf.merge(train.dataset, test.dataset) if test.dataset else train.dataset
    assert "image" in test.test_rendering, "Must have primary image integrator"
    cfg = test.test_rendering["image"]
    train_cfg = OmegaConf.to_container(train.rendering, resolve=True)
    cfg = OmegaConf.merge(train_cfg, cfg) if cfg else train_cfg
    test.test_rendering["image"] = cfg
    logger.info("Merged config:\n" + OmegaConf.to_yaml(test))
    return test

if __name__ == "__main__":
    main()
