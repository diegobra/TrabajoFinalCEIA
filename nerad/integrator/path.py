import drjit as dr
import mitsuba as mi
import numpy as np

from mytorch.utils.profiling_utils import counter_profiler
from nerad.integrator import register_integrator
from nerad.utils.render_utils import mis_weight

import random


@register_integrator("mypath")
class MyPathTracer(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self._init(**props.get("config").dict)

    def _init(
        self,
        hide_emitters: bool,
        return_depth: bool,
        max_depth: int,
        rr_depth: int,
        **kwargs
    ):
        self.hide_emitters = hide_emitters
        self.return_depth = return_depth

        # max depth
        if max_depth < 0 and max_depth != -1:
            raise Exception(
                "\"max_depth\" must be set to -1 (infinite) or a value >= 0")

        # Map -1 (infinity) to 2^32-1 bounces
        self.max_depth = max_depth if max_depth != -1 else 0xffffffff

        if rr_depth <= 0:
            raise Exception(
                "\"rr_depth\" must be set to a value greater than zero!")
        self.rr_depth = rr_depth

    def sample(self,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               medium: mi.Medium,
               active: mi.Bool):

        ray = mi.Ray3f(dr.detach(ray))
        depth = mi.UInt32(0)
        eta = mi.Float(1)
        result = mi.Spectrum(0)
        throughput = mi.Spectrum(1)
        valid_ray = mi.Mask((~mi.Bool(self.hide_emitters))
                            & dr.neq(scene.environment(), None))

        active = mi.Bool(active)                      # Active SIMD lanes

        # Variables caching information from the previous bounce
        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)
        bsdf_ctx = mi.BSDFContext()

        # Record the following loop in its entirety
        loop = mi.Loop(name="MyPathTracer",
                       state=lambda: (sampler, ray, throughput, result,
                                      eta, depth, valid_ray, prev_si, prev_bsdf_pdf,
                                      prev_bsdf_delta, active))

        # Specify the max. number of loop iterations (this can help avoid
        # costly synchronization when when wavefront-style loops are generated)
        loop.set_max_iterations(self.max_depth)

        while loop(active):
            # Compute a surface interaction that tracks derivatives arising
            # from differentiable shape parameters (position, normals, etc.)
            # In primal mode, this is just an ordinary ray tracing operation.

            si = scene.ray_intersect(ray,
                                     ray_flags=mi.RayFlags.All,
                                     coherent=dr.eq(depth, 0))

            # Get the BSDF, potentially computes texture-space differentials
            bsdf = si.bsdf(ray)

            # ---------------------- Direct emission ----------------------

            em_hit_result = self.emitter_hit(
                scene, throughput, prev_si, prev_bsdf_pdf, prev_bsdf_delta, si)
            result += em_hit_result
            # ---------------------- Emitter sampling ----------------------

            # Should we continue tracing to reach one more vertex?
            active_next = (depth + 1 < self.max_depth) & si.is_valid()

            em_sample_result = self.sample_emitter(
                scene, sampler, throughput, bsdf_ctx, si, bsdf, active_next)
            result += em_sample_result

            # ------------------ Detached BSDF sampling -------------------

            bsdf_sample, bsdf_weight, ray = self.bsdf_sample(
                sampler, active, bsdf_ctx, si, bsdf, active_next)

            # ------ Update loop variables based on current interaction ------

            throughput *= bsdf_weight
            eta *= bsdf_sample.eta
            valid_ray |= active & si.is_valid() & ~mi.has_flag(
                bsdf_sample.sampled_type, mi.BSDFFlags.Null)

            # Information about the current vertex needed by the next iteration
            prev_si = si
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(
                bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # -------------------- Stopping criterion ---------------------

            depth[si.is_valid()] += 1
            # Don't run another iteration if the throughput has reached zero
            throughput_max = dr.max(throughput)
            rr_prob = dr.minimum(throughput_max * eta**2, .95)
            rr_active = depth >= self.rr_depth
            rr_continue = sampler.next_1d() < rr_prob
            throughput[rr_active] *= dr.rcp(dr.detach(rr_prob))
            active = active_next & (
                ~rr_active | rr_continue) & dr.neq(throughput_max, 0)

        aov = [depth] if self.return_depth else []
        if counter_profiler.enabled:
            counter_profiler.record("integrator.depth", np.array(depth).tolist())

        return dr.select(valid_ray, result, 0), valid_ray, aov

    def aov_names(self):
        return ['depth'] if self.return_depth else []

    def bsdf_sample(self, sampler, active, bsdf_ctx, si, bsdf, active_next):

        bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                               sampler.next_1d(),
                                               sampler.next_2d(),
                                               active_next)

        # Genera un nuevo rayo a partir de la interacción con la superficie
        ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

        # When the path tracer is differentiated, we must be careful that
        #   the generated Monte Carlo samples are detached (i.e. don't track
        #   derivatives) to avoid bias resulting from the combination of moving
        #   samples and discontinuous visibility. We need to re-evaluate the
        #   BSDF differentiably with the detached sample in that case. */
        if (dr.grad_enabled(ray)):
            ray = dr.detach(ray)

            # Recompute 'wo' to propagate derivatives to cosine term
            wo = si.to_local(ray.d)
            bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, wo, active)
            bsdf_weight[bsdf_pdf > 0] = bsdf_val / dr.detach(bsdf_pdf)

        return bsdf_sample, bsdf_weight, ray

    def emitter_hit(self, scene, throughput, prev_si, prev_bsdf_pdf, prev_bsdf_delta, si):

        # Compute MIS weight for emitter sample from previous bounce
        ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

        mis = mis_weight(
            prev_bsdf_pdf,
            scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
        )

        em_hit_result = throughput * mis * ds.emitter.eval(si)
        return em_hit_result

    import mitsuba as mi

    def get_emission(self, throughput, prev_si, prev_bsdf_pdf, prev_bsdf_delta, si,
                 emitter_pos, emitter_normal, emitter_radius,
                 emitter_radiance=mi.Color3f(12, 17, 4), tolerance=1e-1):
        """
        Retorna la radiancia (Le) para cada interacción en 'si',
        asumiendo un emisor definido manualmente por un disco con centro 'emitter_pos',
        normal 'emitter_normal', y radio 'emitter_radius'.

        Parámetros
        ----------
        scene : Scene
            La escena de Mitsuba.
        throughput : mi.Color3f
            El throughput acumulado hasta la interacción actual.
        prev_si : SurfaceInteraction
            La interacción previa en la trayectoria.
        prev_bsdf_pdf : float
            La densidad de probabilidad del BSDF previa.
        prev_bsdf_delta : bool
            Indica si la interacción previa fue especular (delta BSDF).
        si : SurfaceInteraction (vectorizado)
            Contiene p (posiciones) y n (normales) de muchas interacciones.
        emitter_pos : mi.Vector3f
            Centro del disco emisor.
        emitter_normal : mi.Vector3f
            Normal del disco emisor (no necesariamente normalizada).
        emitter_radius : float
            Radio del disco emisor.
        emitter_radiance : float, mi.Color3f o similar
            Radiancia emitida. Si es un color, puede ser mi.Color3f.
        tolerance : float
            Umbral para la distancia al plano, etc.

        Retorna
        -------
        radiancia : del mismo tipo que 'emitter_radiance'
            Valor de la radiancia ponderada por MIS para cada interacción.
        """

        # p y n pueden ser 'arrays' de dimensión [N, 3] internamente.
        p = si.p
        n = si.n

        # Normalizamos la normal del emisor
        nor = emitter_normal / dr.norm(emitter_normal)

        # Vector d desde el centro hasta cada punto p
        d = p - emitter_pos

        # Distancia (escalar) al plano según la normal
        dist_plano = dr.dot(d, nor)

        # 1. Máscara: verificar que el punto esté cerca del plano
        belongs = dr.abs(dist_plano) < tolerance

        # 2. Comprobar que la proyección en el plano caiga dentro del radio
        d_plano = d - dist_plano * nor
        belongs &= (dr.norm(d_plano) < emitter_radius)

        # 3. Comprobar orientación de la normal de la superficie con la del emisor
        dot_normales = dr.dot(n, nor)

        # 4. Seleccionar Le donde 'belongs' es True, 0 donde es False
        Le_cero = dr.zeros(type(emitter_radiance))  # Construye un "cero" del mismo tipo que Le
        Le = dr.select(belongs, emitter_radiance, Le_cero)

        # Cálculo manual de la probabilidad de dirección del emisor (PDF)
        # Asumiendo un disco emisor uniforme:
        # Área del disco emisor
        emitter_area = dr.pi * emitter_radius ** 2

        # Vector de dirección desde la interacción previa hacia la posición actual
        d_to_emitter = emitter_pos - prev_si.p
        dist_sq = dr.squared_norm(d_to_emitter)
        d_to_emitter = d_to_emitter / dr.sqrt(dist_sq)  # Normalizamos la dirección

        # Cálculo de la PDF direccional manualmente:
        # pdf = (distancia^2) / (área del emisor * cos(θ))
        cos_theta = dr.dot(-d_to_emitter, nor)
        emitter_pdf = dr.select((cos_theta > 0) & ~prev_bsdf_delta,
                                dist_sq / (emitter_area * cos_theta),
                                0.0)

        # Cálculo del peso de Muestras Múltiples Independientes (MIS)
        mis = mis_weight(prev_bsdf_pdf, emitter_pdf)

        # Radiancia final con MIS aplicado
        radiancia = throughput * mis * Le

        return radiancia


    # Sampleo de un punto en el disco del emisor
    def sample_emitter_point(self, sampler, sel_emitter_pos, sel_emitter_normal, sel_emitter_radius):
        """Genera un punto aleatorio en un emisor circular"""

        # selected_pos     = dr.gather(mi.Point3f, positions, index)
        #     selected_normal  = dr.gather(mi.Vector3f, normals, index)
        #     selected_radius  = dr.gather(mi.Float,    radii, index)
        #     selected_radiance= dr.gather(mi.Color3f,  radiances, index)

        # (emitter_pos, emitter_normal, emitter_radius, _) = emitter

        uv = sampler.next_2d()  # shape = [num_samples, 2]
        r   = sel_emitter_radius * dr.sqrt(uv[0])
        phi = 2 * dr.pi * uv[1]

        x = r * dr.cos(phi)
        y = r * dr.sin(phi)

        # Vector local (x, y, 0) en el plano XY
        local_point = mi.Vector3f(x, y, 0.0)  # shape = [num_samples, 3]

        # Construir un marco local a partir de la normal del emisor
        frame = mi.Frame3f(sel_emitter_normal)  # Se transmite con broadcast

        # Pasar el punto local a coordenadas de mundo
        sampled_point = sel_emitter_pos + frame.to_world(local_point)

        return sampled_point


    def emitter_hit_indirect(self, sampler, scene, bsdf_ctx, throughput, prev_si, prev_bsdf_pdf, prev_bsdf_delta, si,
                    emitters, point_direct_light=False,
                    tolerance=1e-1):
        """
        Retorna la radiancia total directa, incluyendo la emisión del emisor y la luz directa desde otras superficies.
        """

        if point_direct_light:
            # Se asume que el emisor es puntual (para simplificaciones en las pruebas)
            # En este caso, se utiliza el centro del círculo
            (sampled_emitter_pos, sampled_emitter_normal, sampled_emitter_radius, sampled_emitter_radiance) = random.choice(emitters)
        else:

            if len(emitters) == 1:
                (sampled_emitter_pos, sampled_emitter_normal, sampled_emitter_radius, sampled_emitter_radiance) = emitters[0]
                sampled_emitter_pos = self.sample_emitter_point(sampler, sampled_emitter_pos, sampled_emitter_normal, sampled_emitter_radius)
            else:

                sample = sampler.next_1d()
                emitter_index = dr.floor(sample * len(emitters))
                emitter_index = mi.UInt32(dr.clamp(emitter_index, 0, len(emitters) - 1))

                positions, normals, radiuss, radiances = zip(*emitters)

                positions_array = mi.Point3f(np.array([(p[0][0], p[1][0], p[2][0]) for p in positions], dtype=np.float32))
                normals_array = mi.Point3f(np.array([(p[0][0], p[1][0], p[2][0]) for p in normals], dtype=np.float32))
                radius_array = mi.Float(np.array([p[0] for p in radiuss], dtype=np.float32))
                radiances_array = mi.Point3f(np.array([(p[0][0], p[1][0], p[2][0]) for p in radiances], dtype=np.float32))

                selected_pos     = dr.gather(mi.Point3f, positions_array, emitter_index)
                selected_normal  = dr.gather(mi.Point3f, normals_array, emitter_index)
                selected_radius  = dr.gather(mi.Float, radius_array, emitter_index)
                selected_radiances  = dr.gather(mi.Point3f, radiances_array, emitter_index)

                #emitter = random.choice(emitters)
                #(_, sampled_emitter_normal, sampled_emitter_radius, sampled_emitter_radiance) = emitter
                sampled_emitter_pos = self.sample_emitter_point(sampler, selected_pos, selected_normal, selected_radius)
                sampled_emitter_normal = selected_normal
                sampled_emitter_radius = selected_radius
                sampled_emitter_radiance = selected_radiances

        # Normalizamos la normal del emisor
        nor = sampled_emitter_normal / dr.norm(sampled_emitter_normal)

        # Vector d desde el centro hasta cada punto p
        d = si.p - sampled_emitter_pos

        # Distancia (escalar) al plano según la normal
        dist_plano = dr.dot(d, nor)

        # 1. Máscara: verificar que el punto esté cerca del plano
        belongs = dr.abs(dist_plano) < tolerance

        # 2. Comprobar que la proyección en el plano caiga dentro del radio
        d_plano = d - dist_plano * nor
        belongs &= (dr.norm(d_plano) < sampled_emitter_radius)

        # 3. Comprobar orientación de la normal de la superficie con la del emisor
        #dot_normales = dr.dot(n, nor)

        # 4. Seleccionar Le donde 'belongs' es True, 0 donde es False
        Le_cero = dr.zeros(type(sampled_emitter_radiance))  # Construye un "cero" del mismo tipo que Le
        Le = dr.select(belongs, sampled_emitter_radiance, Le_cero)

        # Cálculo de la PDF del emisor
        emitter_area = dr.pi * sampled_emitter_radius ** 2
        d_to_emitter = sampled_emitter_pos - si.p
        dist_sq = dr.squared_norm(d_to_emitter)
        dist_to_emitter = dr.norm(d_to_emitter)
        d_to_emitter = d_to_emitter / dr.sqrt(dist_sq)
        cos_theta = dr.maximum(dr.dot(-d_to_emitter, nor), 1e-4)  # Evitar divisiones por cero
        emitter_pdf = dr.select((cos_theta > 1e-4) & ~prev_bsdf_delta,
                                dr.maximum(dist_sq / (emitter_area * cos_theta), 1e-8),
                                0.0)

        # Cálculo del peso de Muestras Múltiples Independientes (MIS)
        mis = mis_weight(prev_bsdf_pdf, emitter_pdf)

        # Radiancia del emisor con MIS aplicado
        radiancia = throughput * mis * Le

        si_normal = si.n
        mask = (dr.dot(si_normal, d_to_emitter) < 0)
        si_normal = dr.select(mask, -si_normal, si_normal)


        # ---------------------------
        # Cálculo de la luz directa desde otras superficies
        # ---------------------------
        epsilon = dr.maximum(1e-3, 1e-4 * d_to_emitter)
        shadow_ray = mi.Ray3f(
            o=si.p + epsilon * si_normal,
            d=d_to_emitter,
            time=si.time,
            wavelengths=si.wavelengths
        )
        shadow_hit = scene.ray_intersect(shadow_ray)
        valid_light_mask = dr.neq(shadow_hit.t, dr.inf) & (shadow_hit.t > dist_to_emitter - 0.01) #& (dr.dot(si_normal, emitter_normal) < 0)

        if dr.any(valid_light_mask):
            bsdf_val, _ = si.bsdf().eval_pdf(bsdf_ctx, si, si.to_local(d_to_emitter), active=True)
            indirect_radiance = throughput * bsdf_val * sampled_emitter_radiance * cos_theta * dr.rcp(dist_sq)
            #indirect_radiance = bsdf_val * emitter_radiance * cos_theta * dr.rcp(dist_sq)
            radiancia += dr.select(valid_light_mask, indirect_radiance, mi.Color3f(0.0))

        return radiancia

    def emitter_hit_indirect2(self, sampler, scene, bsdf_ctx, throughput, prev_si, prev_bsdf_pdf, prev_bsdf_delta, si,
                    emitter_pos, emitter_normal, emitter_radius,
                    emitter_radiance=mi.Color3f(12, 17, 4), tolerance=1e-1):
        """
        Retorna la radiancia total directa, incluyendo la emisión del emisor y la luz directa desde otras superficies.
        """
        # p y n pueden ser 'arrays' de dimensión [N, 3] internamente.
        p = si.p
        n = si.n

        sampled_emitter_pos = self.sample_emitter_point(sampler, emitter_pos, emitter_normal, emitter_radius)

        # Normalizamos la normal del emisor
        nor = emitter_normal / dr.norm(emitter_normal)

        # Vector d desde el centro hasta cada punto p
        d = p - sampled_emitter_pos

        # Distancia (escalar) al plano según la normal
        dist_plano = dr.dot(d, nor)

        # 1. Máscara: verificar que el punto esté cerca del plano
        belongs = dr.abs(dist_plano) < tolerance

        # 2. Comprobar que la proyección en el plano caiga dentro del radio
        d_plano = d - dist_plano * nor
        belongs &= (dr.norm(d_plano) < emitter_radius)

        # 3. Comprobar orientación de la normal de la superficie con la del emisor
        dot_normales = dr.dot(n, nor)

        # 4. Seleccionar Le donde 'belongs' es True, 0 donde es False
        Le_cero = dr.zeros(type(emitter_radiance))  # Construye un "cero" del mismo tipo que Le
        Le = dr.select(belongs, emitter_radiance, Le_cero)

        # Cálculo de la PDF del emisor
        emitter_area = dr.pi * emitter_radius ** 2
        d_to_emitter = sampled_emitter_pos - si.p
        dist_sq = dr.squared_norm(d_to_emitter)
        dist_to_emitter = dr.norm(d_to_emitter)
        d_to_emitter = d_to_emitter / dr.sqrt(dist_sq)
        cos_theta = dr.maximum(dr.dot(-d_to_emitter, nor), 1e-4)  # Evitar divisiones por cero
        emitter_pdf = dr.select((cos_theta > 1e-4) & ~prev_bsdf_delta,
                                dr.maximum(dist_sq / (emitter_area * cos_theta), 1e-8),
                                0.0)

        # Cálculo del peso de Muestras Múltiples Independientes (MIS)
        mis = mis_weight(prev_bsdf_pdf, emitter_pdf)

        # Radiancia del emisor con MIS aplicado
        radiancia = throughput * mis * Le

        # ---------------------------
        # Cálculo de la luz directa desde otras superficies
        # ---------------------------
        epsilon = dr.maximum(1e-4, 1e-4 * d_to_emitter)
        shadow_ray = mi.Ray3f(
            o=si.p + epsilon * d_to_emitter,
            d=d_to_emitter,
            time=si.time,
            wavelengths=si.wavelengths
        )
        shadow_hit = scene.ray_intersect(shadow_ray)
        valid_light_mask = dr.neq(shadow_hit.t, dr.inf) & (shadow_hit.t > dist_to_emitter - 0.01)

        if dr.any(valid_light_mask):
            bsdf_val, _ = si.bsdf().eval_pdf(bsdf_ctx, si, si.to_local(d_to_emitter), active=valid_light_mask)
            indirect_radiance = throughput * bsdf_val * emitter_radiance * cos_theta * dr.rcp(dist_sq)
            radiancia += dr.select(valid_light_mask, indirect_radiance, mi.Color3f(0.0))

        return radiancia

    def emitter_hit_indirect_old_180325(self, scene, bsdf_ctx, sampler, throughput, prev_si, prev_bsdf_pdf, prev_bsdf_delta, si,
                    emitter_pos, emitter_normal, emitter_radius,
                    emitter_radiance=mi.Color3f(12, 17, 4), tolerance=1e-1):
        """
        Retorna la radiancia total directa, incluyendo la emisión del emisor y la luz directa desde otras superficies.
        """
        # p y n pueden ser 'arrays' de dimensión [N, 3] internamente.
        p = si.p
        n = si.n

        sampled_emitter_pos = self.sample_emitter_point(sampler, emitter_pos, emitter_normal, emitter_radius)

        # Vector d desde el centro hasta cada punto p
        d = p - sampled_emitter_pos

        # Distancia (escalar) al plano según la normal
        dist_plano = dr.dot(d, emitter_normal)

        # 1. Máscara: verificar que el punto esté cerca del plano
        belongs = dr.abs(dist_plano) < tolerance

        # 2. Comprobar que la proyección en el plano caiga dentro del radio
        d_plano = d - dist_plano * emitter_normal
        belongs &= (dr.norm(d_plano) < emitter_radius)

        # 3. Comprobar orientación de la normal de la superficie con la del emisor
        dot_normales = dr.dot(n, emitter_normal)

        # 4. Seleccionar Le donde 'belongs' es True, 0 donde es False
        Le_cero = dr.zeros(type(emitter_radiance))  # Construye un "cero" del mismo tipo que Le
        Le = dr.select(belongs, emitter_radiance, Le_cero)

        # Cálculo de la PDF del emisor
        emitter_area = dr.pi * emitter_radius ** 2
        d_to_emitter = sampled_emitter_pos - prev_si.p
        dist_sq = dr.squared_norm(d_to_emitter)
        dist_prevsi_to_emitter = dr.norm(d_to_emitter)
        d_to_emitter = d_to_emitter / dist_prevsi_to_emitter

        #belongs &= dr.dot(-d_to_emitter, emitter_normal) > 0

        cos_theta_emitter = dr.maximum(dr.dot(-d_to_emitter, emitter_normal), 1e-4)

        prev_si_normal = prev_si.n
        mask = (dr.dot(prev_si_normal, d_to_emitter) < 0)
        prev_si_normal = dr.select(mask, -prev_si_normal, prev_si_normal)

        cos_theta_prev = dr.maximum(dr.dot(d_to_emitter, prev_si_normal), 1e-4)

        d_si_to_prevsi = prev_si.p - si.p
        d_si_to_prevsi = d_si_to_prevsi / (dr.norm(d_si_to_prevsi) + 0.01)

        si_normal = si.n
        mask = dr.dot(si_normal, d_si_to_prevsi) < 0
        si_normal = dr.select(mask, -si_normal, si_normal)

        # Cálculo de cosenos correctos

        cos_theta_si = dr.maximum(dr.dot(d_si_to_prevsi, si_normal), 1e-4)

        cos_theta = dr.maximum(dr.dot(-d_to_emitter, emitter_normal), 1e-4)  # Evitar divisiones por cero
        emitter_pdf = dr.select((cos_theta_emitter > 1e-4) & ~prev_bsdf_delta,
                                dr.maximum(dist_sq / (emitter_area * cos_theta_emitter), 1e-8),
                                0.0)

        # Cálculo del peso de Muestras Múltiples Independientes (MIS)
        mis = mis_weight(prev_bsdf_pdf, emitter_pdf)

        # Radiancia del emisor con MIS aplicado
        radiancia = throughput * mis * Le

        # ---------------------------
        # Cálculo de la luz directa desde otras superficies
        # ---------------------------
        epsilon = dr.maximum(1e-4, 1e-4 * dist_prevsi_to_emitter)
        shadow_ray = mi.Ray3f(
            o=prev_si.p + epsilon * d_to_emitter,
            d=d_to_emitter,
            time=prev_si.time,
            wavelengths=prev_si.wavelengths
        )
        shadow_hit = scene.ray_intersect(shadow_ray)
        #valid_light_mask = dr.neq(shadow_hit.t, dr.inf)
        valid_light_mask = dr.neq(shadow_hit.t, dr.inf) & (shadow_hit.t >= (dist_prevsi_to_emitter - 0.1)) & (dr.dot(-d_to_emitter, emitter_normal) > 0)

        # floor_mask = si.p.y < 0.5

        # debug_mask = floor_mask & ~valid_light_mask
        # # O si quieres inspeccionar tanto los que pasan como los que no pasan:
        # # debug_mask = floor_mask

        # # Imprimimos la distancia de shadow_hit, la dist al emisor y el dot con la normal
        # print("debug: shadow_hit.t={}, dist_emit={}, dot={}, val_mask={}, floor_mask={}",
        #         shadow_hit.t[debug_mask],
        #         dist_prevsi_to_emitter[debug_mask],
        #         dr.dot(-d_to_emitter, emitter_normal)[debug_mask],
        #         valid_light_mask[debug_mask],
        #         floor_mask[debug_mask])

        if dr.any(valid_light_mask):
            bsdf_val, _ = si.bsdf().eval_pdf(bsdf_ctx, si, si.to_local(d_si_to_prevsi), active=valid_light_mask)
            #bsdf_val = prev_si.bsdf().eval(bsdf_ctx, prev_si, prev_si.to_local(d_to_emitter), active=valid_light_mask)
            indirect_radiance = throughput * bsdf_val * emitter_radiance * cos_theta_prev * cos_theta_si * dr.rcp(dist_sq)
            radiancia += dr.select(valid_light_mask, indirect_radiance, mi.Color3f(0.0))
            #radiancia = dr.select(valid_light_mask, mi.Color3f(70,0,0), mi.Color3f(0.0))

        return radiancia

    def get_emission_old_260225(self, si, emitter_pos, emitter_normal, emitter_radius, emitter_radiance=mi.Color3f(12, 17, 4), tolerance=1e-1):
        """
        Retorna la radiancia (Le) para cada interacción en 'si',
        asumiendo un emisor definido por un disco con centro 'pos_emisor',
        normal 'normal_emisor', y radio 'radio_emisor'.

        Si la interacción no pertenece al emisor, devuelve 0 (o un valor nulo)
        en esa posición de la máscara.

        Parámetros
        ----------
        si : SurfaceInteraction (vectorizado)
            Contiene p (posiciones) y n (normales) de muchas interacciones.
        pos_emisor : mi.Vector3f
            Centro del disco emisor.
        normal_emisor : mi.Vector3f
            Normal del disco emisor (no necesariamente normalizada).
        radio_emisor : float
            Radio del disco emisor.
        Le : float, mi.Color3f o similar
            Radiancia emitida. Si es un color, puede ser mi.Color3f.
        tolerancia : float
            Umbral para la distancia al plano, etc.

        Retorna
        -------
        radiancia : del mismo tipo que 'Le'
            Valor de la radiancia para cada interacción.
            Donde no pertenezca al emisor, se obtiene 0.
        """

        # p y n pueden ser 'arrays' de dimensión [N, 3] internamente.
        p = si.p
        n = si.n

        # Normalizamos la normal del emisor
        nor = emitter_normal / dr.norm(emitter_normal)

        # Vector d desde el centro hasta cada punto p
        d = p - emitter_pos

        # Distancia (escalar) al plano según la normal
        dist_plano = dr.dot(d, nor)

        # 1. Máscara: verificar que el punto esté cerca del plano
        belongs = dr.abs(dist_plano) < tolerance

        # 2. Comprobar que la proyección en el plano caiga dentro del radio
        d_plano = d - dist_plano * nor
        belongs &= (dr.norm(d_plano) < emitter_radius)

        # 3. Comprobar orientación de la normal de la superficie con la del emisor
        dot_normales = dr.dot(n, nor)
        # Si queremos descartar las normales que apunten en sentido opuesto, pedimos > 0
        #belongs &= (dot_normales > 0.0) # SE COMENTÓ ESTO EL 15/04/25, porque las superficies no necesariamente tenían la misma normal que los emisores

        # 4. Seleccionar Le donde 'belongs' es True, 0 donde es False
        #    Si Le es escalar, usamos 0.0. Si es un color (por ej. mi.Color3f),
        #    podemos usar el constructor correspondiente o `dr.zeros(type(Le))`.
        Le_cero = dr.zeros(type(emitter_radiance))  # Construye un "cero" del mismo tipo que Le
        radiancia = dr.select(belongs, emitter_radiance, Le_cero)

        return radiancia


    def sample_emitter(self, scene, sampler, throughput, bsdf_ctx, si, bsdf, active_next):
        # Is emitter sampling even possible on the current vertex?
        active_em = active_next & mi.has_flag(
            bsdf.flags(), mi.BSDFFlags.Smooth)

        # If so, randomly sample an emitter without derivative tracking.
        ds, em_weight = scene.sample_emitter_direction(
            si, sampler.next_2d(), True, active_em)
        active_em &= dr.neq(ds.pdf, 0.0)

        if (dr.grad_enabled(si.p)):
            # Given the detached emitter sample, *recompute* its
            # contribution with AD to enable light source optimization
            ds.d = dr.normalize(ds.p - si.p)
            em_val = scene.eval_emitter_direction(si, ds, active_em)
            em_weight = dr.select(dr.neq(ds.pdf, 0), em_val / ds.pdf, 0)

            # Evaluate BSDF * cos(theta) differentiably
        wo = si.to_local(ds.d)
        bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
        mis_em = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))
        em_sample_result = throughput * mis_em * bsdf_value_em * em_weight

        return em_sample_result

    import mitsuba as mi


    def sample_custom_emitter(
        self,
        scene,
        sampler,
        throughput,          # mi.Color3f (o array de Dr.Jit) con forma [..., 3]
        prev_bsdf_pdf,       # PDF de la dirección saliente en la intersección anterior
        si,                  # SurfaceInteraction actual
        bsdf,                # BSDF en la intersección actual
        bsdf_ctx,            # mi.BSDFContext, para la evaluación de la BSDF
        emitter_position,    # mitsuba.Point3f
        emitter_normal,      # mitsuba.Normal3f (normalizada)
        emitter_radius,      # float (radio del disco emisor)
        emitter_radiance     # mi.Color3f (radiancia del emisor)
    ):
        """
        Genera num_samples muestras sobre un disco emisor, calcula la contribución
        de luz directa con oclusión y la combina vía MIS con la PDF previa (prev_bsdf_pdf).
        """
        # ---------------------------
        # 1) Generar muestras en el disco (en coordenadas polares)
        # ---------------------------
        uv = sampler.next_2d()  # shape = [num_samples, 2]
        r   = emitter_radius * dr.sqrt(uv[0])
        phi = 2 * dr.pi * uv[1]

        x = r * dr.cos(phi)
        y = r * dr.sin(phi)

        # Vector local (x, y, 0) en el plano XY
        local_point = mi.Vector3f(x, y, 0.0)  # shape = [num_samples, 3]

        # Construir un marco local a partir de la normal del emisor
        frame = mi.Frame3f(emitter_normal)  # Se transmite con broadcast

        # Pasar el punto local a coordenadas de mundo
        sampled_point = emitter_position + frame.to_world(local_point)
        # shape = [num_samples, 3]

        # ---------------------------
        # 2) Construir rayos de sombra
        # ---------------------------
        to_emitter  = sampled_point - si.p  # [num_samples, 3]
        dist_sq     = dr.sum(to_emitter * to_emitter)  # [num_samples]
        valid_mask  = dist_sq > 1e-1 #0.0
        dist        = dr.sqrt(dist_sq)
        dir_to_emitter = to_emitter * dr.rcp(dist)  # normalizar


        epsilon = dr.maximum(0.01, 1e-4 * dist)
        shadow_ray = mi.Ray3f(
            o           = si.p + dir_to_emitter * epsilon, # Se suma este epsilon en dirección hacia el emisor para evitar casos en que intersecta la propia superficie
            d           = dir_to_emitter,
            time        = si.time,
            wavelengths = si.wavelengths
        )

        # ---------------------------
        # 3) Consultar visibilidad
        # ---------------------------
        shadow_hit    = scene.ray_intersect(shadow_ray)
        blocked_mask  = (shadow_hit.t < (dist - 0.1))  # si hay colisión antes, está ocluido

        # ---------------------------
        # 4) Cálculo de la contribución directa
        # ---------------------------

        # 4.1) Factor geométrico "lado emisor"
        # cos_emitter = n_light . (-dir_to_emitter), con la normal apuntando "hacia afuera"
        #cos_emitter = dr.dot(emitter_normal, -dir_to_emitter)

        # Área del disco
        disk_area = dr.pi * (emitter_radius**2)

        # PDF de muestrear un punto en área -> PDF en sólido:
        # p_w = pA * (dist^2 / cos_emitter)   (si cos_emitter > 0)
        # donde pA = 1 / disk_area
        #emitter_pdf_solid_angle = (dist_sq * (1.0 / disk_area)) * dr.rcp(dr.maximum(cos_emitter, 1e-8))

        cos_emitter = dr.maximum(dr.dot(emitter_normal, -dir_to_emitter), 0.0)
        disk_area = dr.pi * (emitter_radius ** 2)
        #emitter_pdf_solid_angle = (dist_sq / disk_area) * dr.rcp(cos_emitter + 1e-4)
        emitter_pdf_solid_angle = (dist_sq / disk_area) * dr.rcp(dr.maximum(cos_emitter, 1e-4))


        #emitter_pdf_solid_angle /= dr.maximum(dr.sum(emitter_pdf_solid_angle), 1.0)


        # 4.2) Peso MIS (balance heuristic)
        #   w = prev_bsdf_pdf / (prev_bsdf_pdf + emitter_pdf_solid_angle)
        #w_mis = mis_weight(prev_bsdf_pdf, emitter_pdf_solid_angle)
        w_mis = mis_weight(emitter_pdf_solid_angle, prev_bsdf_pdf)


        # 4.3) Evaluar la BSDF en la superficie receptora:
        #      "wo" = dirección saliente = la que apunta hacia la luz
        wo = si.to_local(dir_to_emitter)
        bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, wo, active=valid_mask)

        # 4.4) Factor de atenuación geométrica y energía
        inv_dist_sq = dr.rcp(dist_sq)

        # Contribución final de cada sample:
        # throughput * [BSDF] * [radiancia_luz] * [MIS] * [cos_emitter / dist^2]
        contrib = throughput * bsdf_val * emitter_radiance * w_mis * cos_emitter * inv_dist_sq
        #contrib = throughput * bsdf_val * emitter_radiance * cos_emitter * inv_dist_sq


        # ---------------------------
        # 5) Máscara final
        # ---------------------------
        final_mask = valid_mask & (cos_emitter > 0.0) & ~blocked_mask
        contrib = dr.select(final_mask, contrib, mi.Color3f(0.0))

        # ---------------------------
        # 6) Combinar muestras
        # ---------------------------
        # Promedio final (si quieres un único valor por píxel)
        #c_mean = dr.sum(contrib) * (1.0 / float(len(contrib[0])))

        return contrib

    def sample_custom_emitter_indirect(
        self,
        scene,
        sampler,
        throughput,
        prev_bsdf_pdf,
        si,
        bsdf,
        bsdf_ctx,
        emitter_position,
        emitter_normal,
        emitter_radius,
        emitter_radiance
    ):
        """
        Genera muestras sobre un disco emisor y calcula la contribución de luz indirecta mediante un segundo rebote.
        """

        # ---------------------------
        # 1) Generar muestras en el disco emisor
        # ---------------------------
        uv = sampler.next_2d()
        r   = emitter_radius * dr.sqrt(uv[0])
        phi = 2 * dr.pi * uv[1]

        x = r * dr.cos(phi)
        y = r * dr.sin(phi)

        local_point = mi.Vector3f(x, y, 0.0)
        frame = mi.Frame3f(emitter_normal)
        sampled_point = emitter_position + frame.to_world(local_point)

        # ---------------------------
        # 2) Construir rayos de sombra
        # ---------------------------
        to_emitter  = sampled_point - si.p
        dist_sq     = dr.sum(to_emitter * to_emitter)
        valid_mask  = dist_sq > 1e-8
        dist        = dr.sqrt(dist_sq)
        dir_to_emitter = to_emitter * dr.rcp(dist)

        # ✅ Considerar la normal del emisor
        valid_mask &= dr.dot(dir_to_emitter, emitter_normal) > 0

        epsilon = dr.maximum(1e-4, 1e-4 * dist)
        shadow_ray = mi.Ray3f(
            o           = si.p + dir_to_emitter * epsilon,
            d           = dir_to_emitter,
            time        = si.time,
            wavelengths = si.wavelengths
        )

        # ---------------------------
        # 3) Consultar visibilidad
        # ---------------------------
        shadow_hit = scene.ray_intersect(shadow_ray)
        blocked_mask = dr.neq(shadow_hit.t, dr.inf) & (shadow_hit.t < (dist * 0.999))

        # ---------------------------
        # 4) Iluminación indirecta con muestreo correcto
        # ---------------------------
        contrib = mi.Color3f(0.0)
        if dr.any(blocked_mask):
            # ✅ Muestreamos una nueva dirección de rebote basada en la BSDF
            new_wo, bsdf_weight = bsdf.sample(bsdf_ctx, shadow_hit, sampler.next_1d(), sampler.next_2d())

            secondary_ray = mi.Ray3f(
                o = shadow_hit.p + shadow_hit.n * epsilon,
                d = shadow_hit.to_world(new_wo.wo),
                time = shadow_hit.time,
                wavelengths = shadow_hit.wavelengths
            )

            second_hit = scene.ray_intersect(secondary_ray)

            # ✅ Ahora, desde `second_hit.p`, verificamos si el emisor es visible
            to_emitter = emitter_position - second_hit.p
            dist_sq = dr.sum(to_emitter * to_emitter)
            dist = dr.sqrt(dist_sq)
            dir_to_emitter = to_emitter * dr.rcp(dist)

            # ✅ Creamos un tercer rayo para verificar si llega sin obstáculos
            visibility_ray = mi.Ray3f(
                o = second_hit.p + dir_to_emitter * epsilon,
                d = dir_to_emitter,
                time = second_hit.time,
                wavelengths = second_hit.wavelengths
            )

            # ✅ Verificamos si este rayo llega sin obstáculos
            visibility_hit = scene.ray_intersect(visibility_ray)
            visible_mask = dr.neq(visibility_hit.t, dr.inf) & (visibility_hit.t < dist * 0.999)

            # ✅ Si el emisor es visible desde `second_hit.p`, agregamos contribución
            if dr.any(visible_mask):
                cos_theta = dr.maximum(dr.dot(emitter_normal, -dir_to_emitter), 1e-4)
                contrib = throughput * bsdf_weight * emitter_radiance * cos_theta * dr.rcp(dist_sq)

        # ---------------------------
        # 5) Filtrar valores inválidos
        # ---------------------------
        valid_contrib = dr.isfinite(contrib) & dr.any(contrib > 0)
        contrib = dr.select(valid_contrib, contrib, mi.Color3f(0.0))

        return contrib



    def sample_custom_emitter_indirect2(
        self,
        scene,
        sampler,
        throughput,
        prev_bsdf_pdf,
        si,
        bsdf,
        bsdf_ctx,
        emitter_position,
        emitter_normal,
        emitter_radius,
        emitter_radiance
    ):
        """
        Genera muestras sobre un disco emisor, calcula la contribución
        de luz indirecta a partir del segundo rebote.
        """
        # ---------------------------
        # 1) Generar muestras en el disco emisor
        # ---------------------------
        uv = sampler.next_2d()
        r   = emitter_radius * dr.sqrt(uv[0])
        phi = 2 * dr.pi * uv[1]

        x = r * dr.cos(phi)
        y = r * dr.sin(phi)

        local_point = mi.Vector3f(x, y, 0.0)
        frame = mi.Frame3f(emitter_normal)
        sampled_point = emitter_position + frame.to_world(local_point)

        # ---------------------------
        # 2) Construir rayos de sombra
        # ---------------------------
        to_emitter  = sampled_point - si.p
        dist_sq     = dr.sum(to_emitter * to_emitter)
        valid_mask  = dist_sq > 1e-1
        dist        = dr.sqrt(dist_sq)
        dir_to_emitter = to_emitter * dr.rcp(dist)

        epsilon = dr.maximum(0.01, 1e-4 * dist)
        shadow_ray = mi.Ray3f(
            o           = si.p + dir_to_emitter * epsilon,
            d           = dir_to_emitter,
            time        = si.time,
            wavelengths = si.wavelengths
        )

        # ---------------------------
        # 3) Consultar visibilidad
        # ---------------------------
        shadow_hit    = scene.ray_intersect(shadow_ray)
        blocked_mask  = dr.neq(shadow_hit.t, dr.inf) & (shadow_hit.t < (dist - 0.001))

        # ---------------------------
        # 4) Iluminación indirecta (segundo rebote)
        # ---------------------------
        contrib = mi.Color3f(0.0)
        if dr.any(blocked_mask):
            # En vez de lanzar un rayo directo al emisor, muestreamos una nueva dirección aleatoria
            new_dir = bsdf.sample(bsdf_ctx, shadow_hit)
            new_ray = mi.Ray3f(
                o = shadow_hit.p + shadow_hit.n * epsilon,
                d = new_dir,
                time = shadow_hit.time,
                wavelengths = shadow_hit.wavelengths
            )

            # Revisamos si este nuevo rayo llega al emisor
            second_hit = scene.ray_intersect(new_ray)
            valid_secondary = dr.neq(second_hit.t, dr.inf) & second_hit.is_emitter()

            second_hit = scene.ray_intersect(secondary_ray)
            secondary_dist_sq = dr.maximum(dr.sum(secondary_ray.d * secondary_ray.d), 1e-8)
            safe_dist = dr.maximum(dr.sqrt(secondary_dist_sq), 1e-4)
            second_blocked_mask = dr.neq(second_hit.t, dr.inf) & (second_hit.t < safe_dist - 0.001)

            valid_secondary = blocked_mask & ~second_blocked_mask

            if dr.any(valid_secondary):
                second_cos_emitter = dr.maximum(dr.dot(emitter_normal, -secondary_ray.d), 1e-4)

                bsdf_val, _ = bsdf.eval_pdf(bsdf_ctx, shadow_hit, shadow_hit.to_local(secondary_ray.d), active=valid_secondary)

                contrib = throughput * bsdf_val * emitter_radiance * second_cos_emitter * dr.rcp(secondary_dist_sq)

        valid_contrib = dr.isfinite(contrib) & (contrib >= 0)
        contrib = dr.select(valid_contrib, contrib, mi.Color3f(0.0))

        return contrib

    def get_direct_illumination(
        self,
        scene,
        bsdf_ctx,
        throughput,
        prev_si,
        prev_bsdf_pdf,
        prev_bsdf_delta,
        si,
        emitter_pos,
        emitter_normal,
        emitter_radius,
        emitter_radiance=mi.Color3f(12, 17, 4)
    ):
        """
        Retorna la iluminación directa, considerando solo la contribución de emisores visibles.
        """
        p = si.p
        n = si.n
        nor = emitter_normal / dr.norm(emitter_normal)
        d_to_emitter = emitter_pos - si.p
        dist_sq = dr.squared_norm(d_to_emitter)
        d_to_emitter = d_to_emitter / dr.sqrt(dist_sq)
        cos_theta = dr.maximum(dr.dot(-d_to_emitter, nor), 1e-4)

        # Cálculo de la PDF del emisor
        emitter_area = dr.pi * emitter_radius ** 2
        emitter_pdf = dr.select((cos_theta > 1e-4) & ~prev_bsdf_delta,
                                dr.maximum(dist_sq / (emitter_area * cos_theta), 1e-8),
                                0.0)

        # Cálculo del peso de Muestras Múltiples Independientes (MIS)
        mis = mis_weight(prev_bsdf_pdf, emitter_pdf)

        # ---------------------------
        # Cálculo de la visibilidad
        # ---------------------------
        epsilon = dr.maximum(1e-4, 1e-4 * dr.sqrt(dist_sq))
        shadow_ray = mi.Ray3f(
            o=si.p + d_to_emitter * epsilon,  # Pequeño desplazamiento para evitar auto-intersección
            d=d_to_emitter,
            time=si.time,
            wavelengths=si.wavelengths
        )
        shadow_hit = scene.ray_intersect(shadow_ray)
        blocked_mask = (shadow_hit.t < dr.sqrt(dist_sq) - 1e-4)  # Detecta si hay obstrucción antes del emisor
        valid_light_mask = ~blocked_mask  # Iluminación solo si no hay bloqueo

        direct_illumination = mi.Color3f(0.0)
        if dr.any(valid_light_mask):
            bsdf_val, _ = si.bsdf().eval_pdf(bsdf_ctx, si, si.to_local(d_to_emitter), active=valid_light_mask)
            direct_illumination = throughput * bsdf_val * emitter_radiance * cos_theta * dr.rcp(dist_sq)
            direct_illumination = dr.select(valid_light_mask, direct_illumination, mi.Color3f(0.0))

        return direct_illumination

    def emitter_hit_area_light_many_samples_old(
        self,
        scene,
        sampler,
        throughput,          # mi.Color3f (o array Dr.Jit con forma [.., 3])
        prev_bsdf_pdf,       # float, PDF en la intersección anterior
        si,                  # Un SurfaceInteraction (posiblemente vectorizado)
        emitter_position,    # mitsuba.Point3f
        emitter_normal,      # mitsuba.Normal3f (normalizado)
        emitter_radius,      # float
        emitter_radiance    # mi.Color3f
    ):

        # ---------------------------
        # 1) Generar muestras en el disco
        # ---------------------------
        # Tomamos num_samples pares (u, v)
        uv = sampler.next_2d()          # shape = [num_samples, 2]
        r   = emitter_radius * dr.sqrt(uv[0])   # [num_samples]
        phi = 2 * dr.pi * uv[1]                # [num_samples]

        x = r * dr.cos(phi)
        y = r * dr.sin(phi)

        # Creamos un vector local (x, y, 0) en el plano XY
        local_point = mi.Vector3f(x, y, 0.0)       # shape = [num_samples, 3]

        # Construir un marco local a partir de la normal del emisor
        frame = mi.Frame3f(emitter_normal)         # solo uno (broadcast)

        # Pasar el punto local a coordenadas de mundo
        sampled_point = emitter_position + frame.to_world(local_point)
        # shape = [num_samples, 3]

        # ---------------------------
        # 2) Armar rayos de sombra
        # ---------------------------
        to_emitter = sampled_point - si.p  # shape = [num_samples, 3] (broadcast si.p si es escalar)
        dist_sq    = dr.sum(to_emitter * to_emitter)  # shape = [num_samples]

        # Evitar división por cero si la distancia es 0
        valid_mask = dist_sq > 0.0
        dist       = dr.sqrt(dist_sq)
        dir_to_emitter = to_emitter * dr.rcp(dist)  # normalizar

        # Construimos un array de Ray3f para todos los samples
        # (cada campo se "empaqueta" con la misma forma)
        shadow_ray = mi.Ray3f(
            o           = si.p,  # [num_samples, 3]
            d           = dir_to_emitter,                   # [num_samples, 3]
            time        = si.time,
            wavelengths = si.wavelengths
        )

        # ---------------------------
        # 3) Consultar visibilidad
        # ---------------------------
        shadow_hit = scene.ray_intersect(shadow_ray)
        blocked_mask = (shadow_hit.t < dist)  # si hay colisión antes de "dist", está bloqueado

        # ---------------------------
        # 4) Calcular contribución
        # ---------------------------
        # cos_emitter = dot(normal_emisor, -dir_to_emitter)
        cos_emitter = dr.dot(emitter_normal, -dir_to_emitter)

        # PDF de muestrear un punto sobre el área del disco
        disk_area   = dr.pi * (emitter_radius**2)
        emitter_pdf = 1.0 / disk_area

        # Peso MIS: combina pdf de BSDF y pdf del emisor
        # (ejemplo de "balance heuristic": mis_weight = prev_bsdf_pdf / (prev_bsdf_pdf + emitter_pdf))
        mis_weight_val = mis_weight(prev_bsdf_pdf, emitter_pdf)

        # Contribución geométrica: cos / dist^2
        inv_dist_sq = dr.rcp(dist_sq)
        contrib = throughput * emitter_radiance * (cos_emitter * mis_weight_val) * inv_dist_sq
        # shape = [num_samples, 3]

        # Anular contribución si no es válido, si está bloqueado, o si cos <= 0
        final_mask = valid_mask & (cos_emitter > 0) & ~blocked_mask
        contrib = dr.select(final_mask, contrib, mi.Color3f(0.0))

        # ---------------------------
        # 5) Combinar (promediar) las muestras
        # ---------------------------
        # c_mean = dr.mean(contrib, axis=0)  # ->  un mi.Color3f final

        # Si deseas la media
        #c_mean = dr.sum(contrib, axis=0) * (1.0 / float(num_samples))

        return contrib


    def sample_point_on_disk(self, sampler, emitter_position, emitter_normal, emitter_radius):
        """
        Devuelve (sampled_point, sampled_normal, pdf)
        - sampled_point : mitsuba.Point3f
        - sampled_normal: mitsuba.Normal3f (igual a emitter_normal, normal unitaria del disco)
        - pdf           : densidad de prob. de muestrear ese punto (1 / área del disco)
        """

        # 1) Crear un marco local a partir de la normal del emisor
        frame = mi.Frame3f(emitter_normal)

        # 2) Tomar muestra aleatoria
        uv = sampler.next_2d()  # (u, v) en [0,1)
        r = emitter_radius * dr.sqrt(uv[0])
        phi = 2 * dr.pi * uv[1]

        # 3) Obtener coordenadas locales (x, y) y elevación z=0 en el marco local
        x = r * dr.cos(phi)
        y = r * dr.sin(phi)
        local_point = mi.Vector3f(x, y, 0.0)

        # 4) Llevar a coordenadas de mundo
        world_point = emitter_position + frame.to_world(local_point)

        # 5) La normal del disco (ya normalizada)
        sampled_normal = emitter_normal

        # 6) PDF uniforme en el área del disco: 1 / (π * R^2)
        disk_area = dr.pi * (emitter_radius**2)
        pdf = 1.0 / disk_area

        return world_point, sampled_normal, pdf


    def emitter_hit_area_light(
        self,
        scene,                   # La escena Mitsuba
        sampler,                 # Sampler para generar números aleatorios
        throughput,              # Factor de transporte acumulado
        prev_bsdf_pdf,           # PDF de la dirección en la intersección anterior
        si,                      # Intersección actual (mitsuba.SurfaceInteraction3f)
        emitter_position,        # Centro del disco emisor (mitsuba.Point3f)
        emitter_normal,          # Normal del disco emisor (mitsuba.Normal3f)
        emitter_radius,          # Radio del disco
        emitter_radiance         # Radiancia del emisor (mitsuba.Color3f)
    ):

        # 1) Muestrear un punto en el disco
        sampled_point, sampled_normal, emitter_pdf = self.sample_point_on_disk(
            sampler,
            emitter_position,
            emitter_normal,
            emitter_radius
        )

        # Vector desde el punto de intersección hasta el punto del disco
        to_emitter = sampled_point - si.p

        # Distancia y dirección normalizada
        dist_sq = dr.dot(to_emitter, to_emitter)
        if dist_sq == 0.0:
            return mi.Color3f(0.0)

        dist = dr.sqrt(dist_sq)
        dir_to_emitter = to_emitter / dist

        # 2) Rayo de sombra
        shadow_ray = mi.Ray3f(si.p, dir_to_emitter, si.time, si.wavelengths)
        shadow_hit = scene.ray_intersect(shadow_ray)

        # Si golpeamos algo antes de llegar al punto muestreado en el disco, está bloqueado
        if shadow_hit.is_valid() and (shadow_hit.t < dist):
            return mi.Color3f(0.0)

        # 3) Factor geométrico:
        #    cos emisor = dot(normal_disque, -dir_to_emitter)
        #    cos receptor = dot(si.n,  dir_to_emitter)  (si necesitas NdotL en el receptor, depende de la BSDF)
        cos_emitter = dr.dot(sampled_normal, -dir_to_emitter)
        if cos_emitter <= 0.0:
            return mi.Color3f(0.0)

        # 4) Construir la "direction sample" (opcional, si quieres consistencia con Mitsuba)
        ds = mi.DirectionSample3f()
        ds.p = sampled_point
        ds.n = sampled_normal
        ds.d = dir_to_emitter
        ds.dist = dist
        # ds.pdf = emitter_pdf  # (opcional)

        # 5) Calcular peso de Multiple Importance Sampling
        #    - prev_bsdf_pdf: PDF de la dirección elegida en la intersección anterior
        #    - emitter_pdf: PDF de muestrear ESTE punto en el disco
        mis = mis_weight(prev_bsdf_pdf, emitter_pdf)  # Definir tu función mis_weight

        # 6) Contribución final
        #    Formula típica:
        #    radiancia * cos_emitter / dist^2 * (1 / emitter_pdf)
        #    (además multiplicado por throughput, mis, etc.)
        #    Dependiendo de tu BSDF, también podrías necesitar cos_receptor u otros factores.
        contribution = (
            throughput *
            emitter_radiance *
            cos_emitter / dist_sq *
            mis
        )

        # (Si la BSDF en el receptor requiere factor cos(si.n, dir),
        #  podrías multiplicarlo también: cos_receptor = dr.dot(si.n, dir_to_emitter).)

        return contribution


    def emitter_hit_custom(self, scene, throughput, prev_bsdf_pdf, si, emitter_position, emitter_normal, emitter_radius, emitter_radiance):
        """
        Calcula la contribución de un emisor circular en un punto de intersección si.

        :param scene: Escena en la que se está trazando el rayo.
        :param throughput: Factor de transporte acumulado en la trayectoria.
        :param prev_si: Intersección anterior en la trayectoria del rayo.
        :param prev_bsdf_pdf: Probabilidad de la dirección del rayo en la intersección anterior.
        :param prev_bsdf_delta: Indica si la intersección anterior fue en una superficie especular.
        :param si: Intersección actual donde se evalúa la iluminación del emisor.
        :param emitter_position: Posición del emisor (mitsuba.Point3f).
        :param emitter_normal: Normal de la superficie emisora (mitsuba.Normal3f).
        :param emitter_radius: Radio del emisor (float).
        :param emitter_radiance: Radiancia emitida por la fuente (mitsuba.Color3f).

        :return: Contribución del emisor en el punto de intersección si.
        """

        # Vector desde la intersección hasta el centro del emisor
        to_emitter = emitter_position - si.p

        # Cálculo de la distancia sin usar .norm()
        distance_sq = to_emitter.dot_(to_emitter)
        if distance_sq <= 0.0:
            return False

        distance = dr.sqrt(distance_sq)

        # Normalizar el vector dirección hacia el emisor
        direction = to_emitter / distance

        shadow_ray = mi.Ray3f(si.p, direction, si.time, si.wavelengths)
        shadow_hit = scene.ray_intersect(shadow_ray)

        if shadow_hit.t < distance:
            return mi.Color3f(0.0)  # Hay un objeto antes del emisor, no contribuye

        # Proyectar el punto en el plano del emisor para verificar si está dentro del radio
        #projected_distance = mi.dot(to_emitter, emitter_normal)  # Componente normal
        #projected_point = si.p + projected_distance * emitter_normal  # Proyección en el plano
        #in_radius = mi.norm(projected_point - emitter_position) <= emitter_radius  # Verificación radial

        #if not in_radius:
        #    return mi.Color3f(0.0)  # Si está fuera del radio del emisor, no contribuye

        # Cálculo del ángulo entre la normal del emisor y la dirección del punto
        cos_theta = mi.dot(-direction, emitter_normal)
        if cos_theta <= 0.0:
            return mi.Color3f(0.0)  # Si la luz no apunta hacia afuera, no contribuye

        # Crear una muestra de dirección similar a DirectionSample3f
        ds = mi.DirectionSample3f()
        ds.p = emitter_position  # Posición del emisor
        ds.n = emitter_normal  # Normal del emisor
        ds.d = direction  # Dirección hacia la intersección
        ds.dist = distance  # Distancia desde el emisor hasta el punto de intersección

        # Calcular peso de muestreo múltiple por importancia (MIS)
        emitter_pdf = 1 / (mi.pi * emitter_radius**2)  # Aproximación de la densidad de probabilidad del emisor
        mis = mis_weight(prev_bsdf_pdf, emitter_pdf)

        # Evaluar la contribución del emisor
        em_hit_result = throughput * mis * emitter_radiance * cos_theta / (distance**2)

        return em_hit_result


    def to_string(self):
        return (
            "MyPathTracer[\n"
            f"    max_depth={self.max_depth},\n"
            f"    rr_depth={self.rr_depth},\n"
            "]"
        )
