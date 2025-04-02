from typing import Union, Tuple
import gc

import drjit as dr
import mitsuba as mi

from nerad.integrator import register_integrator
from nerad.utils.render_utils import mis_weight
from tqdm import tqdm

import numpy as np

@register_integrator("highquality")
class HighQuality(mi.SamplingIntegrator):
    """
    This class takes care of rendering an image, but instead of rendering the whle pixels all at once, just rendering small blocks at a time and iteratively repeate the same for all blocks until the whole image is rendererd.
    Integrator: mi.SamplingIntegrator
        This is the integrator that takes care of rendering the blocks.
    """
    def __init__(self, block_size: int, integrator: mi.SamplingIntegrator):
        super().__init__(mi.Properties())
        self.block_size = block_size
        self.integrator = integrator
        self.emitter_params = None

    def set_emitter(self, emitter_pos, emitter_normal, emitter_radius, emitter_radiance):
        """Almacena los parámetros del emisor para usarlos en render()."""
        self.emitter_params = (emitter_pos, emitter_normal, emitter_radius, emitter_radiance)

    def prepare(self,
                sensor: mi.Sensor,
                block: mi.ImageBlock,
                seed: int = 0,
                spp: int = 0,
                ):
        """
        This method is another implementation of method pepare in mitsuba/src/python/python/ad/common.py
        The difference is that it prepares the sampler to sample for a wavefront size that matches the block size not the film size.
        """

        film_on_sensor = sensor.film()
        sampler = sensor.sampler()

        if spp != 0:
            sampler.set_sample_count(spp)

        spp = sampler.sample_count()
        sampler.set_samples_per_wavefront(spp)

        block_size = block.size()

        if film_on_sensor.sample_border():
            film_size += 2 * film_on_sensor.rfilter().border_size()
            raise NotImplemented()

        wavefront_size = dr.prod(block_size) * spp

        if wavefront_size > 2**32:
            raise Exception(
                "The total number of Monte Carlo samples required by this "
                "rendering task (%i) exceeds 2^32 = 4294967296. Please use "
                "fewer samples per pixel or render using multiple passes."
                % wavefront_size)

        sampler.seed(seed, wavefront_size)

        return sampler, spp



    def render(self,
               scene: mi.Scene,
               sensor: Union[int, mi.Sensor] = 0,
               seed: int = 0,
               spp: int = 0,
               develop: bool = True,
               evaluate: bool = True,
               weighted_sampling = True) -> mi.TensorXf:

        """
        This method is another implementation of method render() in mitsuba/src/python/python/ad/common.py
        The difference is that it breaks down the rendering task into smaller comutations of blocks in the image instead of rendering it all at one pass.
        It uses the underlying integrator to render each block.
        """

        if isinstance(sensor, int):
            sensor= scene.sensors()[sensor]

        # Se obtienen los emisores circulares
        circular_emitters = self.get_circular_emitters(scene)

        if len(circular_emitters) > 0:
            emitter_pos, emitter_normal, emitter_radius, emitter_radiance = circular_emitters[0]
        else:
            if self.emitter_params is not None:
                (emitter_pos, emitter_normal, emitter_radius, emitter_radiance) = self.emitter_params
            else:

                # # Luz izquierda
                # emitter_pos = mi.Point3f(-1., 1., -0.2)
                # emitter_normal = mi.Point3f(1., 0. , 0.)
                # emitter_radius = mi.Float(0.10)
                # emitter_radiance = mi.Color3f(17.,12.,4.)

                # # Luz derecha
                # emitter_pos = mi.Point3f(1., 1., -0.2)
                # emitter_normal = mi.Point3f(-1., 0. , 0.)
                # emitter_radius = mi.Float(0.10)
                # emitter_radiance = mi.Color3f(17.,12.,4.)

                # Luz del techo
                emitter_pos = mi.Point3f(0.0, 2.0, -0.03)
                emitter_normal = mi.Point3f(0.0, -1.0, 0.0)
                emitter_radius = mi.Float(0.10)
                emitter_radiance = mi.Color3f(17., 12., 4.)

                # # Luz encima de cubo grande
                # emitter_pos = mi.Point3f(-0.3318, 1.2000, -0.3061)
                # emitter_normal = mi.Point3f(0.0, 1.0, 0.0)
                # emitter_radius = mi.Float(0.10)
                # emitter_radiance = mi.Color3f(17., 12., 4.)

                # # Luz livingroom
                # emitter_pos = mi.Point3f(-2.5709776878356934, 2.722001314163208, -1.4513362646102905)
                # emitter_normal = mi.Point3f(0.0, -1.0, 0.0)
                # emitter_radius = mi.Float(0.10)
                # emitter_radiance = mi.Color3f(17., 12., 4.)

        #Prepare the spiral
        spiral = mi.Spiral(sensor.film().crop_size(), mi.ScalarVector2i(0,0), self.block_size)
        sensor.film().prepare(self.integrator.aov_names())
        has_aov = len(self.integrator.aov_names())>0

        direct_light_mask = self.compute_direct_light_mask(scene, sensor, self.integrator, spp, emitter_pos, emitter_normal, emitter_radius)
        direct_light_mask = self.extract_soft_shadow_edges_opencv(direct_light_mask)

        height, width = direct_light_mask.shape

        bitmap = mi.Bitmap(direct_light_mask, pixel_format=mi.Bitmap.PixelFormat.Y)
        bitmap = bitmap.convert(component_format=mi.Struct.Type.UInt8)
        bitmap.write("direct_light_mask.png")



        for i in tqdm(range(spiral.block_count())):
            block_offset, block_size, block_id = spiral.next_block()
            # Prepare an ImageBlock as specified by the film and block size
            block = sensor.film().create_block(block_size)
            block.set_offset(block_offset)

            # Disable derivatives in all of the following
            with dr.suspend_grad():
                # Prepare the film and sample generator for rendering
                sampler, spp = self.prepare(
                    sensor=sensor,
                    block=block,
                    seed=(seed+1)*block_id,
                    spp=spp)
                # Generate a set of rays starting at the sensor
                if weighted_sampling:
                    ray, weight, pos = self.sample_rays_weighted(scene, sensor, sampler, block, direct_light_mask)
                    sampler.seed((seed+1)*block_id, len(ray.o[0]))
                else:
                    ray, weight, pos = self.sample_rays(scene, sensor, sampler, block)

                # Launch the Monte Carlo sampling process in primal mode
                if issubclass(type(self.integrator), mi.ad.common.ADIntegrator):
                    L, valid, _ = self.integrator.sample(
                        mode=dr.ADMode.Primal,
                        scene=scene,
                        sampler=sampler,
                        ray=ray,
                        depth=mi.UInt32(0),
                        δL=None,
                        state_in=None,
                        reparam=None,
                        active=mi.Bool(True),
                        emitter_pos = emitter_pos,
                        emitter_normal = emitter_normal,
                        emitter_radius = emitter_radius,
                        emitter_radiance = emitter_radiance
                    )
                else:
                    L, valid, aov = self.integrator.sample(
                        scene,
                        sampler,
                        ray,
                        None,
                        active = mi.Bool(True),
                        emitter_pos = emitter_pos,
                        emitter_normal = emitter_normal,
                        emitter_radius = emitter_radius,
                        emitter_radiance = emitter_radiance)

                # Only use the coalescing feature when rendering enough samples
                #block.set_coalesce(block.coalesce() and spp >= 4)

                # Accumulate into the image block
                alpha = dr.select(valid, mi.Float(1), mi.Float(0))
                if has_aov:
                    #Assumption: weight is always [1.0, 1.0, 1.0]
                    floatLs = [L[0], L[1], L[2], alpha, weight[0]]
                    all_channels = floatLs + aov
                    block.put(pos, all_channels)
                    del aov
                else:
                    block.put(pos, ray.wavelengths, L * weight, alpha)

                sampler.schedule_state()
                dr.eval(block.tensor())

                # Explicitly delete any remaining unused variables
                del ray, weight, pos, L, valid, alpha
                gc.collect()


                # Perform the weight division and return an image tensor
                sensor.film().put_block(block)

        primal_image = sensor.film().develop()
        dr.schedule(primal_image)
        if evaluate:
            dr.eval()
            dr.sync_thread()


        return primal_image

    def sample_rays(
        self,
        scene: mi.Scene,
        sensor: mi.Sensor,
        sampler: mi.Sampler,
        block: mi.ImageBlock
    ) -> Tuple[mi.RayDifferential3f, mi.Spectrum, mi.Vector2f, mi.Float]:
        """
        This method is another implementation of method sample_rays() in mitsuba/src/python/python/ad/common.py
        The difference is that it prepares the samples for a block of rays instead of the whole image plane pixels.
        """

        block_size = block.size()
        rfilter = sensor.film().rfilter()
        border_size = rfilter.border_size()

        if sensor.film().sample_border():
            block_size += 2 * border_size
            raise NotImplementedError()

        spp = sampler.sample_count()

        # Compute discrete sample position
        idx = dr.arange(mi.UInt32, dr.prod(block_size) * spp)

        # Try to avoid a division by an unknown constant if we can help it
        log_spp = dr.log2i(spp)
        if 1 << log_spp == spp:
            idx >>= dr.opaque(mi.UInt32, log_spp)
        else:
            idx //= dr.opaque(mi.UInt32, spp)

        # Compute the position on the image plane
        pos = mi.Vector2i()
        pos.y = idx // block_size[0]
        pos.x = dr.fma(-block_size[0], pos.y, idx)

        if sensor.film().sample_border():
            pos -= border_size
            raise NotImplementedError()

        pos += mi.Vector2i(block.offset())

        # Cast to floating point and add random offset
        pos_f = mi.Vector2f(pos) + sampler.next_2d()

        # Re-scale the position to [0, 1]^2
        scale = dr.rcp(mi.ScalarVector2f(sensor.film().crop_size()))
        #offset = -mi.ScalarVector2f(block.offset()) * scale #TODO: check why this is wrong in the orignial mitsuba.ad.common.py
        pos_adjusted = pos_f * scale

        aperture_sample = mi.Vector2f(0.0)
        if sensor.needs_aperture_sample():
            aperture_sample = sampler.next_2d()

        time = sensor.shutter_open()
        if sensor.shutter_open_time() > 0:
            time += sampler.next_1d() * sensor.shutter_open_time()

        wavelength_sample = 0
        if mi.is_spectral:
            wavelength_sample = sampler.next_1d()


        with dr.resume_grad():
            ray, weight = sensor.sample_ray_differential(
                time=time,
                sample1=wavelength_sample,
                sample2=pos_adjusted,
                sample3=aperture_sample
            )


        # With box filter, ignore random offset to prevent numerical instabilities
        splatting_pos = mi.Vector2f(pos) if rfilter.is_box_filter() else pos_f
        return ray, weight, splatting_pos

    def sample_rays_weighted_old(
        self,
        scene: mi.Scene,
        sensor: mi.Sensor,
        sampler: mi.Sampler,
        block: mi.ImageBlock,
        direct_light_mask: mi.TensorXf
    ) -> Tuple[mi.RayDifferential3f, mi.Spectrum, mi.Vector2f, mi.Float, mi.Mask]:

        block_size = block.size()
        rfilter = sensor.film().rfilter()
        border_size = rfilter.border_size()

        if sensor.film().sample_border():
            block_size += 2 * border_size
            raise NotImplementedError()

        # Parámetros adaptativos
        spp_base = 1
        spp_max = 16

        # Total de muestras a generar
        total_samples = dr.prod(block_size) * spp_max
        idx = dr.arange(mi.UInt32, total_samples)

        # Índices por píxel y muestra dentro del píxel
        pixel_index = idx // spp_max
        spp_index = idx % spp_max

        # Posiciones 2D dentro del bloque
        px = pixel_index % block_size.x
        py = pixel_index // block_size.x

        # Posiciones absolutas
        x0, y0 = block.offset().x, block.offset().y
        abs_px = px + x0
        abs_py = py + y0

        # Acceso a la máscara
        mask_flat = dr.ravel(direct_light_mask)
        mask_width = direct_light_mask.shape[1]
        linear_idx = abs_py * mask_width + abs_px
        mask_values = dr.gather(mi.Float, mask_flat, linear_idx)

        # spp dinámico por píxel
        max_spp_for_pixel = dr.select(mask_values > 0.0, spp_max, spp_base)
        keep = spp_index < max_spp_for_pixel

        # Posición del píxel en la imagen
        pos = mi.Vector2i(px, py) + mi.Vector2i(block.offset())
        pos_f = mi.Vector2f(pos) + sampler.next_2d()

        # Reescalado a coordenadas [0, 1]^2
        scale = dr.rcp(mi.ScalarVector2f(sensor.film().crop_size()))
        pos_adjusted = pos_f * scale

        # Apertura
        aperture_sample = mi.Vector2f(0.0)
        if sensor.needs_aperture_sample():
            aperture_sample = sampler.next_2d()

        # Tiempo
        time = sensor.shutter_open()
        if sensor.shutter_open_time() > 0:
            time += sampler.next_1d() * sensor.shutter_open_time()

        # Longitudes de onda
        wavelength_sample = 0
        if mi.is_spectral:
            wavelength_sample = sampler.next_1d()

        # Generación de rayos
        with dr.resume_grad():
            ray, weight = sensor.sample_ray_differential(
                time=time,
                sample1=wavelength_sample,
                sample2=pos_adjusted,
                sample3=aperture_sample
            )

        # Opción: filtrado de rayos no utilizados (opcional si se usa active=keep)
        null_ray = mi.RayDifferential3f(
            o = dr.zeros(mi.Point3f),
            d = dr.zeros(mi.Vector3f),
            time = mi.Float(0.0),
            wavelengths = mi.Color0f(0.0)
        )

        ray.o = dr.select(keep, ray.o, null_ray.o)
        ray.d = dr.select(keep, ray.d, null_ray.d)
        ray.time = dr.select(keep, ray.time, null_ray.time)
        ray.wavelengths = dr.select(keep, ray.wavelengths, null_ray.wavelengths)
        ray.d_x = dr.select(keep, ray.d_x, null_ray.d_x)
        ray.d_y = dr.select(keep, ray.d_y, null_ray.d_y)

        weight = dr.select(keep, weight, 0.0)

        # Posición de splatting
        splatting_pos = mi.Vector2f(pos) if rfilter.is_box_filter() else pos_f

        return ray, weight, splatting_pos, keep

    def sample_rays_weighted(
        self,
        scene: mi.Scene,
        sensor: mi.Sensor,
        sampler: mi.Sampler,
        block: mi.ImageBlock,
        direct_light_mask: mi.TensorXf
    ) -> Tuple[mi.RayDifferential3f, mi.Spectrum, mi.Vector2f]:

        block_size = block.size()
        rfilter = sensor.film().rfilter()
        border_size = rfilter.border_size()

        if sensor.film().sample_border():
            block_size += 2 * border_size
            raise NotImplementedError()

        # Parámetros adaptativos
        spp_base = 1
        spp_max = 16

        # Total de muestras a generar
        total_samples = dr.prod(block_size) * spp_max
        idx = dr.arange(mi.UInt32, total_samples)

        # Índices por píxel y muestra dentro del píxel
        pixel_index = idx // spp_max
        spp_index = idx % spp_max

        # Posiciones 2D dentro del bloque
        px = pixel_index % block_size.x
        py = pixel_index // block_size.x

        # Posiciones absolutas en la imagen
        x0, y0 = block.offset().x, block.offset().y
        abs_px = px + x0
        abs_py = py + y0

        # Acceso a la máscara aplanada
        mask_flat = dr.ravel(direct_light_mask)
        mask_width = direct_light_mask.shape[1]
        linear_idx = abs_py * mask_width + abs_px
        mask_values = dr.gather(mi.Float, mask_flat, linear_idx)

        # spp dinámico por píxel
        max_spp_for_pixel = dr.select(mask_values > 0.0, spp_max, spp_base)
        keep = spp_index < max_spp_for_pixel

        # Generar coordenadas de muestreo
        pos = mi.Vector2i(px, py) + mi.Vector2i(block.offset())
        pos_f = mi.Vector2f(pos) + sampler.next_2d()

        # Reescalado a [0, 1]^2
        scale = dr.rcp(mi.ScalarVector2f(sensor.film().crop_size()))
        pos_adjusted = pos_f * scale

        # Apertura
        aperture_sample = mi.Vector2f(0.0)
        if sensor.needs_aperture_sample():
            aperture_sample = sampler.next_2d()

        # Tiempo
        time = sensor.shutter_open()
        if sensor.shutter_open_time() > 0:
            time += sampler.next_1d() * sensor.shutter_open_time()

        # Longitudes de onda
        wavelength_sample = 0
        if mi.is_spectral:
            wavelength_sample = sampler.next_1d()

        # Generación de rayos
        with dr.resume_grad():
            ray, weight = sensor.sample_ray_differential(
                time=time,
                sample1=wavelength_sample,
                sample2=pos_adjusted,
                sample3=aperture_sample
            )

        # === Filtrado real ===
        keep_i = dr.select(keep, mi.UInt32(1), mi.UInt32(0))
        prefix = dr.prefix_sum(keep_i)
        count  = dr.sum(keep_i)

        idx_all       = dr.arange(mi.UInt32, len(keep))
        scatter_idx   = prefix
        mask_cond     = keep  # <- ya es un Bool array

        valid_idx_data    = dr.gather(type(idx_all), idx_all, idx_all, mask_cond)
        valid_idx_data = self.unique_values_drjit(valid_idx_data)
        valid_scatter_idx = dr.gather(type(scatter_idx), scatter_idx, idx_all, mask_cond)
        valid_scatter_idx = self.unique_values_drjit(valid_scatter_idx)

        valid_idx = dr.zeros(mi.UInt32, count[0])
        dr.scatter(valid_idx, valid_idx_data, valid_scatter_idx)

        # === Rayos finales ===
        ray_filtered    = dr.gather(mi.RayDifferential3f, ray, valid_idx)
        weight_filtered = dr.gather(type(weight), weight, valid_idx)
        pos_filtered    = dr.gather(type(pos_f), pos_f, valid_idx)

        return ray_filtered, weight_filtered, pos_filtered


    def unique_values_drjit(self, drjit_array):

        # Pasar a numpy (sin gradientes)
        arr_np = np.array(drjit_array, dtype=np.uint32)

        # Usar numpy para obtener los únicos
        unique_np = np.unique(arr_np)

        # Volver a drjit
        unique = mi.UInt32(unique_np)

        return unique

    def to_string(self):
        return (
            "Highquality[\n"

            "]"
        )

    def get_circular_emitters(self, scene):
        """
        Extrae todos los emisores de la escena que sean de tipo área y devuelve su
        posición, normal y radio estimado a partir del área del emisor.
        """
        emitters = []

        for shape in scene.shapes():
            if shape.is_emitter():  # Verificar si la forma tiene un emisor
                emitter = shape.emitter()

                # Obtener el centroide de la forma en espacio de mundo
                position = shape.bbox().center()

                sampled_position = shape.sample_position(0, (0,0))

                # Extraer la normal aproximada a partir de la geometría
                normal = sampled_position.n

                # Estimar el radio del emisor a partir del área
                area = shape.surface_area()
                radius = dr.sqrt(area / dr.pi)  # Radio estimado suponiendo un disco equivalente

                print('Área calculada = ', area)

                print('radius = ', radius)

                radiance = mi.Color3f(17., 12., 4.)

                emitters.append((mi.Point3f(position), normal, radius, radiance))

        return emitters


    def compute_direct_light_mask(self, scene, sensor, integrator, spp, emitter_pos, emitter_normal, emitter_radius):
        spiral = mi.Spiral(sensor.film().crop_size(), mi.ScalarVector2i(0, 0), self.block_size)
        #mask_image = dr.zeros(mi.Float, [sensor.film().crop_size().y, sensor.film().crop_size().x])  # 2D
        height, width = sensor.film().crop_size()
        #mask_image = mi.TensorXf(dr.zeros(mi.Float, height * width)).reshape([height, width])
        mask_image = dr.zeros(mi.TensorXf, (height, width))

        film_size = sensor.film().crop_size()
        global_mask_block = mi.ImageBlock(
            film_size,
            mi.ScalarPoint2i(0),    # offset
            1,                      # channel_count
            sensor.film().rfilter() # reconstruction filter
        )

        global_mask_block.clear()

        for i in range(spiral.block_count()):
            block_offset, block_size, block_id = spiral.next_block()
            block = sensor.film().create_block(block_size)
            block.set_offset(block_offset)

            # Crear un bloque local del mismo tamaño que el bloque actual
            block_mask = mi.ImageBlock(
                block_size,
                block_offset,
                1,
                sensor.film().rfilter()
            )

            block_mask.set_offset(block_offset)

            with dr.suspend_grad():
                sampler, spp = self.prepare(
                    sensor=sensor,
                    block=block,
                    seed=(1) * block_id,
                    spp=spp)
                ray, weight, pos = self.sample_rays(scene, sensor, sampler, block)
                si = scene.ray_intersect(ray)

                # Verificamos si el punto intersecado pertenece al emisor
                sampled_emitter_pos = integrator.sample_emitter_point(sampler, emitter_pos, emitter_normal, emitter_radius)
                nor = emitter_normal / dr.norm(emitter_normal)
                d = si.p - sampled_emitter_pos
                dist_plano = dr.dot(d, nor)
                belongs = dr.abs(dist_plano) < 1e-1
                d_plano = d - dist_plano * nor
                belongs &= (dr.norm(d_plano) < emitter_radius)

                # Si no está en el emisor, vemos si recibe luz directa
                d_to_emitter = sampled_emitter_pos - si.p
                dist_sq = dr.squared_norm(d_to_emitter)
                dist_to_emitter = dr.norm(d_to_emitter)
                d_to_emitter = d_to_emitter / dr.sqrt(dist_sq)

                si_normal = si.n
                mask = (dr.dot(si_normal, d_to_emitter) < 0)
                si_normal = dr.select(mask, -si_normal, si_normal)

                epsilon = dr.maximum(1e-3, 1e-4 * d_to_emitter)
                shadow_ray = mi.Ray3f(
                    o=si.p + epsilon * si_normal,
                    d=d_to_emitter,
                    time=si.time,
                    wavelengths=si.wavelengths
                )
                shadow_hit = scene.ray_intersect(shadow_ray)

                cos_theta = dr.maximum(dr.dot(-d_to_emitter, nor), 1e-4)

                sees_light = dr.neq(shadow_hit.t, dr.inf) & (shadow_hit.t > dist_to_emitter - 0.01) & (cos_theta > 0.01)  # No hay obstáculo => luz directa

                has_direct_light = belongs | sees_light  # Unión: emisor o iluminación directa

                # 4. Valores a escribir: 1.0 si hay luz directa, 0.0 si no
                values = dr.select(has_direct_light, 1.0, 0.0)

                block_mask.put(pos, [values])
                global_mask_block.put_block(block_mask)

        mask_image = global_mask_block.tensor()

        return mask_image

    def extract_soft_shadow_edges_opencv(self, direct_light_mask: mi.TensorXf, thickness: int = 20) -> mi.TensorXf:
        """
        A partir de una máscara binaria, genera una máscara de bordes
        con cierto grosor usando OpenCV.

        Args:
            direct_light_mask: mi.TensorXf (valores 0 o 1)
            thickness: número de píxeles de grosor del borde

        Returns:
            edge_mask: mi.TensorXf (valores 0 o 1), solo bordes ensanchados
        """

        import numpy as np
        import cv2

        # Convertir a array de numpy (uint8 para OpenCV)
        mask_np = np.array(direct_light_mask.numpy(), dtype=np.uint8) * 255

        # Detectar bordes con Canny (más robusto)
        edges = cv2.Canny(mask_np, threshold1=50, threshold2=150)

        # Engrosar bordes usando dilatación
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
        thick_edges = cv2.dilate(edges, kernel)

        # Normalizar a rango [0, 1] y convertir a float32
        thick_edges = (thick_edges > 0).astype(np.float32)

        # Volver a TensorXf
        edge_mask = mi.TensorXf(thick_edges)

        return edge_mask
