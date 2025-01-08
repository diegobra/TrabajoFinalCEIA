import mitsuba as mi
import matplotlib.pyplot as plt
import numpy as np

def render_scene_in_popup(scene: mi.Scene, interaction: mi.SurfaceInteraction3f, integrator_name: str = "direct"):
    """
    Renderiza una escena básica con Mitsuba y superpone puntos de interacción.

    :param scene: Una instancia de mitsuba.Scene que representa la escena a renderizar.
    :param interactions: Lista de SurfaceInteraction[] a mostrar sobre la imagen renderizada.
    :param integrator_name: Nombre del integrador, por defecto "direct" para renderizado básico.
    """
    # 1. Configurar un integrador básico
    integrator = mi.load_dict({"type": integrator_name})

    # 2. Renderizar la escena con menos precisión
    print("Rendering scene with basic settings...")
    img = mi.render(scene, integrator=integrator, spp=1)  # spp=1 reduce el número de muestras por píxel

    # 3. Convertir la imagen a un formato visualizable
    img = np.clip(img, 0.0, 1.0)  # Asegurarse de que los valores estén en [0, 1]

    # 4. Obtener información del sensor
    sensor = scene.sensors()[0]  # Usar el primer sensor de la escena
    film_size = sensor.film().size()  # Tamaño de la película en píxeles
    to_camera = sensor.world_transform().inverse()  # Transformación mundo -> cámara

    interaction_positions = []

    cam_space = to_camera @ interaction.p

    # Normalizar para obtener coordenadas 2D de pantalla
    if cam_space.z > 0:  # Solo proyectar puntos frente a la cámara
        screen_pos = mi.Point2f(cam_space.x / cam_space.z, cam_space.y / cam_space.z)

        # Mapear a coordenadas de píxeles en la película
        pixel_pos = mi.Point2f(
            (screen_pos.x + 1) * 0.5 * film_size.x,
            (screen_pos.y + 1) * 0.5 * film_size.y
        )
        interaction_positions.append(pixel_pos)

    # 5. Mostrar la imagen renderizada con marcadores de interacciones
    plt.figure(figsize=(8, 8))
    plt.imshow(img, origin="lower")
    plt.axis("off")
    plt.title("Renderizado con Interacciones")

    # Dibujar los puntos de interacción en la imagen
    for pos in interaction_positions:
        plt.plot(pos.x, pos.y, 'ro', markersize=5, label="Interaction")

    # Mostrar la imagen con los puntos
    plt.legend()
    plt.show()
