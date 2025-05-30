import drjit as dr
import mitsuba as mi

import numpy as np

class ShapeSampler():
    def __init__(self, scene, no_specular_samples, avoid_emitters=False) -> None:
        self.valid_inds = self.compute_valid_sahpes(scene, no_specular_samples, avoid_emitters)
        self.sampler = mi.load_dict({'type': 'independent'})
        self.PCG = None


    def sample_on_shape(self, active, shape):
        sample_state = self.sampler.wavefront_size()

        if sample_state != len(active):
            print('error in sampler')

        pos_rnd = self.sampler.next_2d(active)
        dir_rnd = self.sampler.next_2d(active)

        # mi.cuda_ad_rgb.ShapePtr
        # print('dir(shape) = ', dir(shape))
        pos_samples = shape.sample_position(time = 0,sample = pos_rnd)

        si = mi.SurfaceInteraction3f(ps = pos_samples,wavelengths = [])
        si.initialize_sh_frame()
        si.shape = shape
        si.t = si.p[0]*0
        si.time = si.p[0]*0


        is_twosided = mi.has_flag(si.shape.bsdf().flags(), mi.BSDFFlags.BackSide)

        dir_samples = mi.warp.square_to_uniform_hemisphere(dir_rnd)

        samplingProb = mi.warp.square_to_uniform_hemisphere_pdf(dir_samples)*pos_samples.pdf
        dir_samples[is_twosided] = mi.warp.square_to_uniform_sphere(dir_rnd)
        samplingProb[is_twosided] = mi.warp.square_to_uniform_sphere_pdf(dir_samples)*pos_samples.pdf

        si.wi = dir_samples

        return si, samplingProb

    def rng(self, n):
        if(self.PCG is None):
            self.PCG = mi.PCG32(size=n)
        return self.PCG

    def sample_input(self, scene, n, seed):
        sample_on_shape = True

        self.sampler.set_sample_count(n)
        self.sampler.seed(seed, n)

        self.sampler.schedule_state()
        dr.eval()

        if (sample_on_shape):

            # Obtiene la lista de formas en la escena
            # Ej. en Cornell Box son 8 incluyendo paredes, techo, piso y fuente de luz (Rectangle)
            # y dos cajas (Cube)
            shapes = scene.shapes_dr()

            # Se obtienen índices random según el tamaño del sampleo (ej. 32768)
            indices = self.sample_valid_shape_indices()

            # Se obtienen la formas correspondientes a los índices sampleados (ej. Rectangle, Cube, Cube, Cube, etc.)
            shape = dr.gather(mi.ShapePtr, shapes, indices)
            # print('El tipo de shape es: ', type(shape))

            # Revisar para qué se hace esto. Ver en qué caso active podría tener algún elemento en False
            active = ~dr.isnan(indices)
            si, prob  = self.sample_on_shape(active, shape)
            #diego.render_scene_in_popup(scene, si)
            to_ret = si
        else:
            pos = mi.Vector3f(self.sampler.next_1d(), self.sampler.next_1d(), self.sampler.next_1d())
            dir = mi.warp.square_to_uniform_sphere(self.sampler.next_2d())
            pos = pos*(scene.bbox().max - scene.bbox().min) + scene.bbox().min

            rays = mi.Ray3f(o = pos, d = dir)
            prob = mi.warp.square_to_uniform_sphere_pdf(dir)
            to_ret = rays

        return to_ret, prob

    def sample_random_emitter(self, scene, seed, channel_dropout=True, total_power = 21.19):
        """Muestra una posición aleatoria en cualquier superficie de la escena y la devuelve como un emisor."""

        self.sampler.set_sample_count(1)
        self.sampler.seed(seed, 1)

        shapes = scene.shapes()
        areas = [shape.surface_area() for shape in shapes]
        area_sum = sum(areas)

        # Construir distribución acumulativa
        probs = [a / area_sum for a in areas]
        cum_probs = np.cumsum(probs)

        # Elegir un shape según su área
        u = self.sampler.next_1d()
        for i, cp in enumerate(cum_probs):
            if u[0] < cp:
                shape = shapes[i]
                break

        # Samplear posición sobre el shape elegido
        sample = shape.sample_position(0, self.sampler.next_2d())
        position = sample.p
        normal = sample.n

        # Aproximar radio como la raíz cuadrada del área del shape dividido por PI
        #area = shape.surface_area()
        #radius = np.sqrt(area / np.pi) if area > 0 else 0.0
        #radius = dr.select(area > 0, dr.sqrt(area / np.pi), 0.0)

        #radius = mi.Float(0.1) # A modo de prueba se define el radio manualmente
        radius = self.sampler.next_1d() * 0.3
        #radius = 0.02 + self.sampler.next_1d() * (0.2 - 0.02) # Se establece un radio mínimo para evitar artefectos en la imagen


        #radiance = mi.Color3f(17.,12.,4.)

        if not channel_dropout:
            radiance = mi.Color3f(self.sampler.next_1d()*20,self.sampler.next_1d()*20,self.sampler.next_1d()*20)
        else:

            # Se realiza una especie de "dropout de canales" para la radiancia del emisor
            # Con esto se quiere probar si forzando apagar diferentes combinaciones de canales se aprende mejor en el entrenamiento

            on_off_patterns = [
                (0, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
                (1, 1, 1),
            ]

            pattern = on_off_patterns[seed % len(on_off_patterns)]
            #pattern = on_off_patterns[mi.UInt32(self.sampler.next_1d()*7)[0]]

            r_rnd = self.sampler.next_1d()
            g_rnd = self.sampler.next_1d()
            b_rnd = self.sampler.next_1d()

            # Generar valores aleatorios por canal, multiplicados por el patrón (0 o 1)
            r = pattern[0] * r_rnd
            g = pattern[1] * g_rnd
            b = pattern[2] * b_rnd

            # Calcular norma euclídea y escalar para obtener magnitud = total_power
            norm = dr.sqrt(r**2 + g**2 + b**2) + 1e-8  # evitar división por cero
            scale = total_power / norm

            radiance = mi.Color3f(r * scale, g * scale, b * scale)


        return position, normal, radius, radiance, shape


    def compute_valid_sahpes(self, scene, no_specular_sample, avoid_emitters=False):
        i = 0
        valid_inds, invalid_inds = [], []
        for sh in scene.shapes():
            try:
                sh.surface_area()
                if not (no_specular_sample & mi.has_flag(sh.bsdf().flags(), mi.BSDFFlags.Delta)) and (avoid_emitters == False or not sh.is_emitter()) :
                    valid_inds.append(i)
            except:
                invalid_inds.append(i)
            i += 1

        valid_inds = mi.UInt32(valid_inds)
        return valid_inds


    def sample_valid_shape_indices(self):
        max_len  = len(self.valid_inds)
        random_indices_in_the_valid_array = dr.minimum(mi.UInt32(self.sampler.next_1d() * max_len), max_len-1)
        indices = dr.gather(mi.UInt32, self.valid_inds, random_indices_in_the_valid_array)
        return indices
