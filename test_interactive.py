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
from nerad.model.sampler import ShapeSampler
from nerad.utils.render_utils import render_and_save_image
from nerad.utils.sensor_utils import create_sensor
from nerad.utils.json_utils import write_json

import time

logger = logging.getLogger(__name__)

import cv2
import numpy as np

import gc

def tensor_to_opencv(outputs):
    """
    Convierte un tensor de Mitsuba a una imagen correctamente normalizada
    para visualizar en OpenCV.
    """
    # Convertir a Bitmap de Mitsuba con la misma configuración usada al guardar PNG
    img_mi = mi.Bitmap(outputs[1].numpy())

    # Convertir a formato sRGB (igual que cuando se guarda como PNG)
    img_mi = img_mi.convert(
        mi.Bitmap.PixelFormat.RGB,  # No necesitamos el canal alfa
        mi.Struct.Type.UInt8,  # Convertir a uint8 para OpenCV
        True  # Aplicar corrección gamma automáticamente
    )

    # Convertir la imagen Bitmap de Mitsuba a NumPy
    img_np = np.array(img_mi)



    # Convertir RGB a BGR para OpenCV
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    return img_np

def get_intersection(scene, sensor, x, y, img_width, img_height):
    """
    Lanza un rayo desde el sensor de Mitsuba para obtener la intersección con la escena.
    """

    # Convertir coordenadas de imagen (x, y) a Normalized Device Coordinates (NDC) [0, 1]
    pos_f = mi.Vector2f(x / img_width, y / img_height)

    # Si la cámara tiene apertura, se requiere una muestra para la apertura
    aperture_sample = mi.Vector2f(0.0)
    if sensor.needs_aperture_sample():
        aperture_sample = mi.Point2f(0,0)

    # Muestrear el tiempo de exposición si la cámara lo soporta
    time = sensor.shutter_open()
    if sensor.shutter_open_time() > 0:
        time += mi.Float(0.) * sensor.shutter_open_time()

    # Muestrear la longitud de onda si Mitsuba está en modo espectral
    wavelength_sample = 0
    if mi.is_spectral:
        wavelength_sample = mi.Float(0.)

    # Generar el rayo correspondiente a la coordenada en la imagen
    ray, _ = sensor.sample_ray_differential(
        time=time,
        sample1=wavelength_sample,
        sample2=pos_f,
        sample3=aperture_sample
    )

    # Consultar la intersección con la escena
    si = scene.ray_intersect(ray)

    if si.is_valid()[0]:

        normal = si.n

        # Determinar si el rayo impactó desde dentro o fuera de la cara
        dot_product = dr.dot(si.n, ray.d)

        if dot_product[0] > 0:  # Si el rayo impactó desde dentro, invertir la normal
            #print("El rayo impactó desde dentro, invirtiendo la normal")
            normal = -normal

        #print(f"Intersección en ({si.p[0]}, {si.p[1]}, {si.p[2]})")
        #print(f"   Normal: ({normal})")
        #print(f"   Albedo: {si.bsdf()}")  # Muestra información del material
        #print(f"   Shape: {si.shape[0].id()}")  # Muestra información del material

        return si.p, normal  # Retorna la posición y la intersección completa
    else:
        #print("No se encontró intersección")
        return None, None

def on_mouse_click(event, x, y, flags, param):
    """
    Callback que se activa cuando el usuario hace clic en la imagen renderizada.
    """
    scene, sensor, img_width, img_height, test_integrators, transforms, parm_img_clicked = param
    #print(f"Click en imagen en ({x}, {y})")

    # Obtener intersección
    position, normal = get_intersection(scene, sensor, x, y, img_width, img_height)

    if position is not None:
        #print(f"Coordenadas en mundo: {position}")

        if event == cv2.EVENT_LBUTTONDOWN:

            parm_img_clicked[0] = True

            emitter_pos = position
            emitter_normal = normal
            emitter_radius = mi.Float(0.10)
            emitter_radiance = mi.Color3f(17.,12.,4.)

            test_integrators['image'].add_emitter(emitter_pos, emitter_normal, emitter_radius, emitter_radiance)


            # shape_sampler = ShapeSampler(scene, no_specular_samples=False)

            # # Generar emisor aleatorio contenido en superficie plana
            # seed = int(time.time() * 1000) % (2**32 - 1)
            # emitter_pos, emitter_normal, emitter_radius, emitter_radiance, _ = shape_sampler.sample_random_emitter(
            #     scene, seed
            # )

            # test_integrators['image'].add_emitter(emitter_pos, emitter_normal, emitter_radius, emitter_radiance)

        elif event == cv2.EVENT_MBUTTONDOWN:
            # Mover y orientar la cámara mirando en dirección contraria a la normal
            forward = -normal / dr.norm(normal)[0]

            # Vector hacia arriba (referencia mundial)
            world_up = mi.Vector3f(0.0, 1.0, 0.0)

            # Evitar degeneración si la normal es vertical
            if abs(dr.dot(forward, world_up)[0]) > 0.99:
                world_up = mi.Vector3f(0.0, 0.0, 1.0)

            right = dr.normalize(dr.cross(world_up, forward))
            up = dr.cross(forward, right)

            # Construir nueva matriz to_world desde los vectores base
            new_to_world = np.eye(4)
            new_to_world[0, :3] = np.array(right)
            new_to_world[1, :3] = np.array(up)
            new_to_world[2, :3] = np.array(forward)
            new_to_world[:3, 3] = np.array(position)

            transforms['0']['to_world'] = new_to_world.tolist()
            parm_img_clicked[0] = True


def interactive_render(cfg_test_rendering, scene, transforms, images, test_integrators, out_root):
    """
    Muestra la imagen renderizada en un popup interactivo y permite navegar la cámara
    con las flechas del teclado en tiempo real.
    """
    view_idx = 0  # Índice inicial de vista

    cv2.namedWindow("Renderizado Interactivo")

    recalculate_image = True
    parm_img_clicked = {0: False}

    last_render_time = 0.0
    fps = 0.0

    img_LHS_np = None
    sensor = None

    if "fov" not in transforms['0']:
        transforms['0']["fov"] = 45.0  # Valor por defecto si no está presente

    current_fov = transforms['0']["fov"]

    while True:

        if recalculate_image:

            start_time = time.time()

            dr.flush_malloc_cache()
            torch.cuda.empty_cache()
            gc.collect()

            #gt = {"image": images[view_idx] if images is not None else None}
            del sensor
            sensor = create_sensor(cfg_test_rendering.width, transforms[str(view_idx)])

            start_timer = time.perf_counter()
            outputs = render_and_save_image(
                out_root / "interactive",
                f"{view_idx:03d}",
                scene,
                test_integrators["image"],
                cfg_test_rendering,
                sensor,
                save_image_to_disk=False
            )
            end_timer = time.perf_counter()

            print(f"Tiempo renderizado:  {end_timer - start_timer:.3f} s")

            del img_LHS_np

            img_LHS_np = tensor_to_opencv(outputs)
            #img_LHS_np = normalize_for_display(outputs[0], clip_max=1.0)
            #img_LHS_np = tone_map_reinhard(outputs[0])
            img_height, img_width, _ = img_LHS_np.shape

            del outputs

            recalculate_image = False
            parm_img_clicked = {0: False}

            end_time = time.time()
            last_render_time = end_time - start_time
            fps = 1.0 / last_render_time if last_render_time > 0 else 0.0

            # Dibujar FPS sobre la imagen
            #cv2.putText(img_LHS_np, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(
                img_LHS_np,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_DUPLEX,   # Fuente más limpia y moderna
                0.7,                       # Tamaño más equilibrado
                (240, 240, 240),           # Color blanco suave
                1,
                cv2.LINE_AA                # Antialiasing activado
            )



        cv2.setMouseCallback("Renderizado Interactivo", on_mouse_click, param=(scene, sensor, img_width, img_height, test_integrators, transforms, parm_img_clicked))

        if parm_img_clicked[0] == True:
            recalculate_image = True

        cv2.imshow("Renderizado Interactivo", img_LHS_np)
        key = cv2.waitKey(100)  # Espera 100ms por una tecla

        if key == 27:  # Tecla 'ESC' para salir
            break
        elif key == 83 or key == 81:  # Flecha derecha o izquierda: rotar sobre el eje Y de la cámara (yaw)
            theta = -2.0 * np.pi / 180 if key == 83 else 2.0 * np.pi / 180
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            # Matriz de rotación alrededor del eje Y local (yaw)
            R_y = np.array([
                [ cos_t, 0, sin_t, 0],
                [     0, 1,     0, 0],
                [-sin_t, 0, cos_t, 0],
                [     0, 0,     0, 1]
            ])

            # Obtener matriz to_world actual
            to_world = np.array(transforms['0']['to_world'])

            # Extraer la posición (columna 3)
            position = to_world[:3, 3].copy()

            # Eliminar la traslación temporalmente
            to_world[:3, 3] = 0

            # Aplicar rotación (local yaw)
            to_world = R_y @ to_world

            # Restaurar la posición original
            to_world[:3, 3] = position

            # Guardar transformación
            transforms['0']['to_world'] = to_world.tolist()
            recalculate_image = True


        elif key == 82 or key == 84:  # Flecha arriba

            # Moverse en la dirección de la cámara
            if key == 82:
                step_size = -0.1  # Distancia a avanzar
            else:
                step_size = 0.1  # Distancia a retroceder

            # Extraer la dirección Z de la cámara
            direction_z = np.array(transforms['0']['to_world'])[2, :3]  # Toma solo los primeros 3 valores (vector dirección)
            to_world = np.array(transforms['0']['to_world'])
            direction_z = -to_world[:3, 2]

            # Normalizar por si acaso (aunque debería ser unitario)
            direction_z = direction_z / np.linalg.norm(direction_z)

            # Modificar la traslación en la dirección de la cámara
            transforms['0']['to_world'][0][3] += step_size * direction_z[0]
            transforms['0']['to_world'][1][3] += step_size * direction_z[1]
            transforms['0']['to_world'][2][3] += step_size * direction_z[2]

            recalculate_image = True

        elif key == ord('w') or key == ord('s'):
            theta = -5 * np.pi / 180 if key == ord('w') else 5 * np.pi / 180
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            # Matriz de rotación alrededor del eje X local de la cámara
            R_x = np.array([
                [1,     0,      0,     0],
                [0, cos_t, -sin_t,     0],
                [0, sin_t,  cos_t,     0],
                [0,     0,      0,     1]
            ])

            # Obtener la matriz to_world actual
            to_world = np.array(transforms['0']['to_world'])

            # Guardar la posición actual
            position = to_world[:3, 3].copy()

            # Eliminar traslación temporalmente
            to_world[:3, 3] = 0

            # Aplicar rotación local (rotar la cámara sobre su eje X)
            to_world = to_world @ R_x

            # Restaurar la posición
            to_world[:3, 3] = position

            transforms['0']['to_world'] = to_world.tolist()
            recalculate_image = True

        elif key == ord('m'):  # aumentar FOV
            current_fov = min(120.0, current_fov + 5.0)
            transforms['0']["fov"] = current_fov
            print(f"Nuevo FOV: {current_fov}")
            recalculate_image = True

        elif key == ord('n'):  # disminuir FOV
            current_fov = max(5.0, current_fov - 5.0)
            transforms['0']["fov"] = current_fov
            print(f"Nuevo FOV: {current_fov}")
            recalculate_image = True

        elif key == ord('i'):
            cfg_test_rendering.only_indirect = not cfg_test_rendering.only_indirect
            test_integrators['image'].set_custom_config(cfg_test_rendering, interactive_test=True)
            recalculate_image = True


        elif key == ord('c'):
            test_integrators['image'].clear_emitters()
            recalculate_image = True

        elif key == ord('+'):
            test_integrators['image'].increase_emitters_radius()
            recalculate_image = True

        elif key == ord('-'):
            test_integrators['image'].decrease_emitters_radius()
            recalculate_image = True

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

    cfg_test_rendering = cfg.test_rendering["image"]
    test_integrators['image'].set_custom_config(cfg_test_rendering, interactive_test=True)
    interactive_render(cfg_test_rendering, scene, transforms, images, test_integrators, out_root)


def merge_config(test: TestConfig, train: TrainConfig) -> TestConfig:
    test.dataset = OmegaConf.merge(train.dataset, test.dataset) if test.dataset else train.dataset


    assert "image" in test.test_rendering, "Must have primary image integrator"

    cfg = test.test_rendering["image"]
    train_cfg = OmegaConf.to_container(train.rendering, resolve=True)
    cfg = OmegaConf.merge(train_cfg, cfg) if cfg else train_cfg

    test.test_rendering["image"] = cfg

    logger.info("Merged config:\n" + OmegaConf.to_yaml(test))
    return test

def normalize_for_display(tensor_img, clip_max=2.0):
    """
    Normaliza el tensor de Mitsuba para visualizar sin que los valores extremos dominen.
    """
    img = tensor_img.numpy()
    img = np.clip(img / clip_max, 0, 1)
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def tone_map_reinhard(tensor_img):
    """
    Aplica tone mapping tipo Reinhard: x / (1 + x)
    Ideal para manejar alto rango dinámico sin definir clip explícito.
    """
    img = tensor_img.numpy()
    img = img / (1.0 + img)
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    main()
