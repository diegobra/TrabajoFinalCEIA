import logging
import os
from pathlib import Path

import drjit as dr
import hydra
import mitsuba as mi
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from tqdm import tqdm

from common import (compute_output_metrics, configure_compute, create_integrator, create_loss_function,
                    load_dataset, prepare_learned_objects)
from nerad.integrator.highquality import HighQuality
from nerad.model.config import ObjectConfig, TestConfig, TrainConfig
from nerad.utils.render_utils import render_and_save_image
from nerad.utils.sensor_utils import create_sensor
from nerad.utils.json_utils import write_json

logger = logging.getLogger(__name__)

import cv2
import numpy as np

def tensor_to_opencv(outputs):
    """
    Convierte un tensor de Mitsuba a una imagen correctamente normalizada
    para visualizar en OpenCV.
    """
    # Convertir a Bitmap de Mitsuba con la misma configuración usada al guardar PNG
    img_mi = mi.Bitmap(outputs[0].numpy())

    # Convertir a formato sRGB (igual que cuando se guarda como PNG)
    img_mi = img_mi.convert(
        mi.Bitmap.PixelFormat.RGB,  # No necesitamos el canal alfa
        mi.Struct.Type.UInt8,  # Convertir a uint8 para OpenCV
        True  # Aplicar corrección gamma automáticamente
    )

    # Convertir la imagen Bitmap de Mitsuba a NumPy
    img_np = np.array(img_mi)

    # Convertir RGB → BGR para OpenCV
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    return img_np

def interactive_render(cfg, scene, transforms, images, test_integrators, out_root):
    """
    Muestra la imagen renderizada en un popup interactivo y permite navegar la cámara
    con las flechas del teclado en tiempo real.
    """
    view_idx = 0  # Índice inicial de vista
    num_views = len(transforms)

    while True:
        dr.flush_malloc_cache()
        torch.cuda.empty_cache()

        gt = {"image": images[view_idx] if images is not None else None}
        rendering = cfg.test_rendering["image"]
        sensor = create_sensor(rendering.width, transforms[str(view_idx)])

        outputs = render_and_save_image(
            out_root / "interactive",
            f"{view_idx:03d}",
            scene,
            test_integrators["image"],
            rendering,
            sensor,
        )

        img_LHS_np = tensor_to_opencv(outputs)

        cv2.imshow("Renderizado Interactivo", img_LHS_np)
        key = cv2.waitKey(100)  # Espera 100ms por una tecla

        if key == 27:  # Tecla 'ESC' para salir
            break
        elif key == ord('d') or key == 83:  # Flecha derecha
            view_idx = (view_idx + 1) % num_views
        elif key == ord('a') or key == 81:  # Flecha izquierda
            view_idx = (view_idx - 1) % num_views
        elif key == ord('w') or key == 82:  # Flecha arriba
            pass  # Implementar movimiento en eje Y si es necesario
        elif key == ord('s') or key == 84:  # Flecha abajo
            pass  # Implementar movimiento en eje Y si es necesario

    cv2.destroyAllWindows()


@hydra.main(version_base="1.2", config_path="config", config_name="test")
def main(cfg: TestConfig = None):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    use_hq = True

    # print('cfg (1) = ', cfg)

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

        integrator = create_integrator(rendering, scene, cfg.dataset.scene, post_init_injection=integrator_injection,
                                       kwargs_injection=integrator_function_injection)

        test_integrators[key] = integrator
    logger.info(f"Integrators:\n{test_integrators}")

    # Load checkpoint
    learned_info = prepare_learned_objects(
        scene,
        test_integrators["image"],
        learned_modules,
        None,
        train_root / "checkpoints" / f"{cfg.ckpt}.ckpt",
        device,
    )

    if use_hq:
        block_size = cfg.blocksize
        logger.info(f"High quality renderer being used at block size: {block_size}")
        test_integrators = {
            k: HighQuality(block_size, v) for k, v in test_integrators.items()
        }

    view_indices = cfg.views
    if len(view_indices) == 0:
        n_views = 1 if cfg.n_views <= 0 else cfg.n_views
        view_indices = list(range(n_views))

    # Esto es nuevo. Agregar controles para su ejecución o quitarlo de test.py y ejecutarlo aparte. 090325
    interactive_render(cfg, scene, transforms, images, test_integrators, out_root)

    logger.info(f"Render {len(view_indices)} views to {out_root}")
    all_metrics = {}
    for idx in tqdm(view_indices):
        dr.flush_malloc_cache()
        torch.cuda.empty_cache()

        gt = {
            "image": images[idx] if images is not None else None,
        }
        view_metrics = {}

        for name, rendering in test_rendering.items():
            sensor = create_sensor(rendering.width, transforms[str(idx)])


            outputs = render_and_save_image(
                out_root / name,
                f"{idx:03d}",
                scene,
                test_integrators[name],
                rendering,
                sensor,
            )

            metrics = compute_output_metrics(name, outputs, rendering.integrator, gt)
            view_metrics.update(metrics)

        logger.info(f"Metrics for view {idx}\n" +
                    "\n".join((f"{k:>20} {v:.6f}" for k, v in view_metrics.items())) + "\n")
        all_metrics[str(idx)] = view_metrics

    write_json(out_root / "metrics.json", all_metrics)


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
