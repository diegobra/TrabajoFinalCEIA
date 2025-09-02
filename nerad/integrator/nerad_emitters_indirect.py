import drjit as dr
import mitsuba as mi
import torch
import torch.nn as nn

from nerad.integrator import register_integrator
from nerad.loss import LossFucntion
from nerad.mitsuba_wrapper import wrapper_registry
from nerad.model.sampler import ShapeSampler
from nerad.texture.dictionary import MiDictionary

from .path import MyPathTracer

from .nerad import Nerad

import numpy as np

import time
from datetime import datetime

@register_integrator("nerad_emitters_indirect")
class NeradEmittersIndirect(Nerad, nn.Module):
    def __init__(self, props: mi.Properties):
        nn.Module.__init__(self)
        MyPathTracer.__init__(self, props)
        self.residual_sampler = None
        self.residual_sampler_m = None
        self.residual_function = None
        self.network = None
        self.return_only_LHS = props.get("config").dict.get("return_only_LHS")
        self.use_autocast_rhs = props.get("config").dict.get("use_autocast_rhs")
        self.m = props.get("config").dict.get("m")

        self.emitter_pos_train = None
        self.emitter_normal_train = None
        self.emitter_radius_train = None
        self.emitter_radiance_train = None

    def post_init(
        self,
        residual_function: LossFucntion,
        function: str,
        kwargs: MiDictionary,
    ):
        self.residual_function = residual_function
        self.network = wrapper_registry.build(function, kwargs)

    def get_albedo_detached(self, si):
        with dr.suspend_grad():
            with torch.no_grad():
                reflect = si.bsdf().eval_diffuse_reflectance(si)
        return reflect

    def compute_residual(self, scene, n, seed):
        if self.residual_sampler is None:
            self.residual_sampler = ShapeSampler(scene, no_specular_samples=True, avoid_emitters=True)
            assert self.m != 0
            self.residual_sampler_m = self.residual_sampler.sampler.clone()
            self.residual_sampler_m.seed(seed,n*self.m)

        self.residual_sampler_m.schedule_state()
        si, _ = self.residual_sampler.sample_input(scene=scene, n=n, seed=seed)

        self.emitter_pos_train, self.emitter_normal_train, self.emitter_radius_train, self.emitter_radiance_train, shape_emitter_train = self.residual_sampler.sample_random_emitter(scene, seed)
        # print('self.emitter_pos_train = ', self.emitter_pos_train)
        # print('self.emitter_normal_train = ', self.emitter_normal_train)
        # print('self.emitter_radius_train = ', self.emitter_radius_train)
        # print('shape_emitter.id() = ', shape_emitter_train.id())
        # print('------------------------')

        # # Luz izquierda
        # self.emitter_pos_train = mi.Point3f(-1., 1., -0.2)
        # self.emitter_normal_train = mi.Point3f(1., 0. , 0.)
        # self.emitter_radius_train = mi.Float(0.10)
        # self.emitter_radiance_train = mi.Color3f(17.,12.,4.)

        # if seed % 3 == 0:
        #     # Luz izquierda
        #     self.emitter_pos_train = mi.Point3f(-1., 1., -0.2)
        #     self.emitter_normal_train = mi.Point3f(1., 0. , 0.)
        #     self.emitter_radius_train = mi.Float(0.10)
        #     self.emitter_radiance_train = mi.Color3f(17.,12.,4.)
        # elif seed % 3 == 1:
        #     # Luz derecha
        #     self.emitter_pos_train = mi.Point3f(1., 1., -0.2)
        #     self.emitter_normal_train = mi.Point3f(-1., 0. , 0.)
        #     self.emitter_radius_train = mi.Float(0.10)
        #     self.emitter_radiance_train = mi.Color3f(17.,12.,4.)
        # elif seed % 3 == 2:
        #     # Luz del techo
        #     self.emitter_pos_train = mi.Point3f(0., 2.0, -0.03)
        #     self.emitter_normal_train = mi.Point3f(0.0, -1.0, 0.0)
        #     self.emitter_radius_train = mi.Float(0.10)
        #     self.emitter_radiance_train = mi.Color3f(17., 12., 4.)

        emitters_train = [(self.emitter_pos_train, self.emitter_normal_train, self.emitter_radius_train, self.emitter_radiance_train)]

        _, _, aov = self.sample(scene, self.residual_sampler.sampler, si, 0, True,
                                emitters_train,
                                sampler_m = self.residual_sampler_m)
        residual = mi.Color3f(aov[-3:])
        return residual

    def log(self, mensaje):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {mensaje}")

    def sample(self,
               scene: mi.Scene,         # Información de la escena
               sampler: mi.Sampler,     # Generador de números aleatorios
               ray: mi.Ray3f,           # Rayos sampleados en la escena (sobre las diferentes formas)
                                        # Recibe SurfaceInteraction cuando está entrenando y Ray3f cuando rederiza la imagen
               medium: mi.Medium,
               active: mi.Bool,
               emitters,
               point_direct_light=False,
               return_only_direct_light=False,
               force_return_only_LHS=False,
               use_autocast_lhs=False,
               **kwargs):

        m = 1

        (emitter_pos, emitter_normal, emitter_radius, emitter_radiance) = emitters[0]
        sampler_m = kwargs.get("sampler_m", None)
        if sampler_m is not None:
            m = self.m # 32

        depth = mi.UInt32(0)        # [0]
        eta = mi.Float(1)           # [1.0]
        throughput = mi.Spectrum(1) # [1.0, 1.0, 1.0]

        valid_ray = mi.Mask((~mi.Bool(self.hide_emitters))
                            & dr.neq(scene.environment(), None)) # [False]

        active = mi.Bool(active)  # Active SIMD lanes # [True]

        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = mi.Float(1.0) # [1,0]
        prev_bsdf_delta = mi.Bool(True) # [True]
        bsdf_ctx = mi.BSDFContext()

        if isinstance(ray, mi.SurfaceInteraction3f):
            # ray tiene tipo mi.SurfaceInteraction3f en el entrenamiento
            si = ray
            bsdf = si.bsdf()
            #Assertion: there shouldn't be ANY specualr sample here assuming that the code gets here only when residual sampling
            assert (dr.none(mi.has_flag(si.bsdf().flags(), mi.BSDFFlags.Delta)))
        else:
            # Tiene tipo Ray3f en el renderizado de la imagen
            ray = mi.Ray3f(dr.detach(ray))
            si = scene.ray_intersect(ray,
                                     ray_flags=mi.RayFlags.All,
                                     coherent=dr.eq(depth, 0))

            bsdf = si.bsdf(ray)


        if return_only_direct_light:
            E = self.emitter_hit_indirect(sampler, scene, bsdf_ctx, throughput, prev_si, prev_bsdf_pdf, prev_bsdf_delta,
                                dr.detach(si), emitters, point_direct_light=point_direct_light, include_emitter_radiance=True)

            mask = valid_ray | (active & si.is_valid())
            LHS = dr.select(mask, E, 0)
            zero_vec = LHS*0
            return zero_vec, mask, [LHS.x, LHS.y, LHS.z, dr.select(mask, mi.Float(1), mi.Float(0)), zero_vec.x, zero_vec.y, zero_vec.z]

        # En bsdf queda una descripción detallada de la BSDF para cada interacción (ej. tipo de reflexión)
        # Ej. para una interacción particular:
            # TwoSided[
            #     brdf[0] = SmoothDiffuse[
            #         reflectance = SRGBReflectanceSpectrum[
            #         value = [[0.14, 0.45, 0.091]]
            #         ]
            #     ],
            #     brdf[1] = SmoothDiffuse[
            #         reflectance = SRGBReflectanceSpectrum[
            #         value = [[0.14, 0.45, 0.091]]
            #         ]
            # ]

        # ---------------------- Handle specular surfaces (if any) ----------------------

        # Esta función sólo se utiliza en nerad_specular. Para nerad no specular, simplemente retorna los mismos valores de entrada
        si, prev_si, prev_bsdf_pdf, prev_bsdf_delta, valid_ray, throughput, eta = self.trace_speculars(scene, sampler, si, active, prev_si, prev_bsdf_pdf, prev_bsdf_delta, valid_ray, throughput, eta)
        bsdf = si.bsdf() # No parece necesario

        # ---------------------- Eval LHS ----------------------
        # A partir de si (surface interaction) se obtienen los parámetros de entrada a la red neuronal
        pts, dirs, normals, albedo = self.extract_inputs(si)

        # Se evalúa la radiosidad para cada interacción en la red neuronal
        # self.network.set_use_autocast_rhs(use_autocast_lhs) # Se usaría autocast en LHS sólo para inferencia. Para entrenamiento no funciona bien con backpropagation
        LHS = dr.select(active & si.is_valid(), self.network.eval(pts, dirs, normals, albedo, emitter_pos, emitter_normal, emitter_radius, emitter_radiance), mi.Vector3f(0))

        # Se calcula LHS para los rayos válidos
        LHS = dr.select(active & si.is_valid(), throughput*LHS, mi.Vector3f(0))

        # ---------------------- Direct emission ----------------------

        # E = self.emitter_hit_indirect(sampler, scene, bsdf_ctx, throughput, prev_si, prev_bsdf_pdf, prev_bsdf_delta,
        #                        dr.detach(si), emitters, point_direct_light=point_direct_light)

        if self.return_only_LHS or force_return_only_LHS:
           mask = valid_ray | (active & si.is_valid())
           LHS = dr.select(mask, LHS, 0)
           zero_vec = LHS*0
           return zero_vec, mask, [LHS.x, LHS.y, LHS.z, dr.select(mask, mi.Float(1), mi.Float(0)), zero_vec.x, zero_vec.y, zero_vec.z]


        # ---------------------- repeat (if requested) ----------------------
        if m > 1:
            # Se preparan las variables para el sampleo de montecarlo de RHS
            indices = dr.arange(mi.UInt, 0, len(si.p[0])) # [0, 1, 2, 3, 4, .. 32758 skipped .., 32763, 32764, 32765, 32766, 32767]
            indices = dr.repeat(indices, self.m) # [0, ... (32 veces) ..., 0, 1, ..., 1, ... ... , 32767, ... , 32767] tamaño 32768*32
            si = dr.gather(type(si), si, indices) # Interacciones para cada índice (tamaño 32768*32)
            prev_bsdf_delta = dr.gather(type(prev_bsdf_delta), prev_bsdf_delta, indices)
            prev_bsdf_pdf = dr.gather(type(prev_bsdf_pdf), prev_bsdf_pdf, indices)
            prev_si = dr.gather(type(si), prev_si, indices)
            throughput = dr.gather(type(throughput), throughput, indices)
            eta = dr.gather(type(eta), eta, indices)
            valid_ray = dr.gather(type(valid_ray), valid_ray, indices)


            bsdf = si.bsdf()
            sampler = sampler_m

        # ---------------------- Emitter sampling ----------------------

        active_next = si.is_valid() # Tamaño 32768*32

        # ------------------ Detached BSDF sampling -------------------

        bsdf_sample, bsdf_weight, ray = self.bsdf_sample(
            sampler, active, bsdf_ctx, si, bsdf, active_next)

        # ------ Update loop variables based on current interaction ------

        throughput *= bsdf_weight
        eta *= bsdf_sample.eta
        valid_ray |= active & si.is_valid() & ~mi.has_flag(
            bsdf_sample.sampled_type, mi.BSDFFlags.Null)

        prev_si = si
        prev_bsdf_pdf = bsdf_sample.pdf
        prev_bsdf_delta = mi.has_flag(
            bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

        # -------------------- Stopping criterion ---------------------

        # Si depth es 0 indica que es la primera intersección
        depth[si.is_valid()] += 1 # depth = [1, 1, 1, .. 1048566 skipped .. ,1, 1, 1]
        active = active_next # active_next = [True, True, True, .. 1048566 skipped .. ,True, True, True]

        # Se intersecta cada uno de los 1078566 rayos sampleados con la escena
        si = scene.ray_intersect(ray,
                                 ray_flags=mi.RayFlags.All,
                                 coherent=dr.eq(depth, 0))

        # Se dtermina la bsdf para cada intersección en la dirección del rayo
        bsdf = si.bsdf(ray)

        # ---------------------- Handle specular surfaces (if any) ----------------------

        # Si no hay superficies especulares se devuelve lo mismo que se le pasa por parámetro
        si, prev_si, prev_bsdf_pdf, prev_bsdf_delta, valid_ray, throughput, eta = self.trace_speculars(scene, sampler, si, active, prev_si, prev_bsdf_pdf, prev_bsdf_delta, valid_ray, throughput, eta)
        bsdf = si.bsdf()

        # ---------------------- Direct emission ----------------------

        bsdf_sample_result = self.emitter_hit_indirect(sampler, scene, bsdf_ctx, throughput, prev_si, prev_bsdf_pdf, prev_bsdf_delta,
                                               dr.detach(si), emitters)

        # ---------------------- Eval RHS ----------------------
        with dr.suspend_grad():
            with torch.no_grad():
                pts, dirs, normals, albedo = self.extract_inputs(si)
                if self.use_autocast_rhs:
                    self.network.set_use_autocast_rhs(True)
                RHS_net = dr.select(active & si.is_valid(),
                                    self.network.eval(pts, -ray.d, normals, albedo, emitter_pos, emitter_normal, emitter_radius, emitter_radiance), mi.Vector3f(0))
                if self.use_autocast_rhs:
                    self.network.set_use_autocast_rhs(False)

        #RHS = RHS_net * throughput + bsdf_sample_result + em_sample_result
        #RHS = RHS_net * throughput + em_sample_result
        RHS = RHS_net * throughput + bsdf_sample_result

        # ---------------------- Deal with repeat (if any) ----------------------
        if m > 1:
            RHS = dr.block_sum(RHS, self.m)/self.m
            validity = dr.select(valid_ray, mi.Float(1), mi.Float(0))
            valid_ray = dr.block_sum(validity, self.m)>0

        # aov = dr.select(valid_ray, E + LHS, 0)
        # rgb = dr.select(valid_ray, E + RHS, 0)

        #aov = dr.select(valid_ray, E + bsdf_sample_result, 0)

        aov = dr.select(valid_ray, LHS, 0)
        rgb = dr.select(valid_ray, RHS, 0)

        residual = dr.select(valid_ray, self.residual_function.compute_loss(LHS, RHS), 0)

        return rgb, valid_ray, [aov.x, aov.y, aov.z, dr.select(valid_ray, mi.Float(1), mi.Float(0)), residual.x, residual.y, residual.z]

    def extract_inputs_old(self, si):
        pts = si.p # Puntos de interacción con la escena
        dirs = si.to_world(si.wi)   # Direcciones para cada punto (se hace to_world para convertir desde el
                                    # sistema de referencia local a la superficie al sistema de referencia global)
        normals = si.sh_frame.n # sh_frame es el shading frame, el sistema de referencia local
        normals = dr.select(dr.dot(dirs, normals)<0, -normals, normals) # Se determina hacia dónde apunta la normal
                                                                        # tomando en cuenta el rayo de salida
        albedo = dr.detach(self.get_albedo_detached(si)) # Medida de reflectancia difusa de la superficie en el punto de interacción
        return pts,dirs,normals,albedo

    def extract_inputsgfds(self, si, active=None):
        pts = si.p
        dirs = si.to_world(si.wi)
        normals = si.sh_frame.n
        normals = dr.select(dr.dot(dirs, normals) < 0, -normals, normals)
        albedo = dr.detach(self.get_albedo_detached(si))

        if active is not None:
            idx = dr.arange(mi.UInt32, len(active))
            filtered_idx = idx[active]

            prueba = self.gather_filtered(si, active)

            pts     = dr.gather(type(pts), pts, filtered_idx)
            dirs    = dr.gather(type(dirs), dirs, filtered_idx)
            normals = dr.gather(type(normals), normals, filtered_idx)
            albedo  = dr.gather(type(albedo), albedo, filtered_idx)

        return pts, dirs, normals, albedo

    def gather_filtered(self, array, mask):
        # 1. Convertir mask a enteros
        mask_i = dr.select(mask, dr.scalar.UInt32(1), dr.scalar.UInt32(0))

        # 2. Prefix sum
        prefix = dr.prefix_sum(mask_i)

        # 3. Cantidad de válidos
        count = dr.sum(mask_i)

        # 4. Índices globales
        idx_all = dr.arange(dr.scalar.UInt32, len(mask))
        scatter_idx = prefix - 1

        # 5. Filtrar sólo los datos válidos
        mask_cond = dr.neq(mask_i, dr.scalar.UInt32(0))
        valid_scatter_idx = dr.gather(type(scatter_idx), scatter_idx, mask_cond)
        valid_idx_data = dr.gather(type(idx_all), idx_all, mask_cond)

        # 6. Scatter a buffer final
        valid_idx = dr.zeros(dr.scalar.UInt32, count)
        dr.scatter(valid_idx, valid_idx_data, valid_scatter_idx)

        # 7. Gather final
        return dr.gather(type(array), array, valid_idx)




    def aov_names(self):
        return ["LHS.R", "LHS.G", "LHS.B", "LHS.a", "residual.x", "residual.y", "residual.z"]

    def to_string(self):
        return (
            "NeradIntegrator[\n"
            f"  network={self.network}\n"
            f"  residual_function={self.residual_function}\n"
            "]"
        )

    def traverse(self, callback):
        self.network.traverse(callback)


    def trace_speculars(self,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               active: mi.Bool,
               prev_si: mi.SurfaceInteraction3f,
               prev_bsdf_pdf: mi.Float,
               prev_bsdf_delta: mi.Bool,
               valid_ray: mi.Bool,
               throughput: mi.Spectrum,
               eta: mi.Float
               ):
                #Implemented in the child Nerad Specular
        return ray, prev_si, prev_bsdf_pdf, prev_bsdf_delta, valid_ray, throughput, eta

    def is_specular(self, si):
            return si.is_valid() & False

    def get_emitter_test(self, scene):
        """
        Obtiene un emisor de forma aleatorio de la escena y devuelve su posición central,
        normal y un radio que abarque su área.

        Parámetros:
            scene (mi.Scene): Escena de Mitsuba.

        Retorna:
            tuple: (posición (numpy array), normal (numpy array), radio (float))
        """
        # Obtener todos los emisores de la escena
        emitters = [shape for shape in scene.shapes() if shape.is_emitter()]

        if not emitters:
            raise ValueError("No hay emisores en la escena.")

        # Seleccionar un emisor aleatorio
        emitter = np.random.choice(emitters)

        # Obtener el bounding box del emisor
        bbox = emitter.bbox()

        # Calcular el centro del bounding box (posición representativa del emisor)
        position = (bbox.min + bbox.max) / 2

        # Evaluar un punto en la superficie del emisor para obtener la normal
        sample = emitter.sample_position(0.5, [0.5, 0.5])  # Muestra un punto en el emisor
        normal = sample.n  # Normal de la superficie en ese punto

        # Calcular un radio que cubra la forma entera
        radius = np.linalg.norm(bbox.extents()) / 2

        position = mi.Point3f(position.x, position.y, position.z)
        radius = mi.Float(radius)

        return position, normal, radius
