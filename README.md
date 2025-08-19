# Simulación de Iluminación Global con Redes Neuronales

Este repositorio implementa un sistema de simulación de **iluminación global en escenas tridimensionales mediante redes neuronales**, desarrollado en el marco del Trabajo Final de la *Carrera de Especialización en Inteligencia Artificial (FIUBA)*.

El proyecto toma como **base el repositorio [Neural Radiosity](https://saeedhd96.github.io/neural-radiosity/)** de *Hadadan et al. (SIGGRAPH Asia 2021)*, extendiéndolo con nuevas funcionalidades y adaptaciones que permiten:  

- Incorporar **emisores variables** como entrada a la red neuronal.  
- Entrenar la red para estimar **únicamente la iluminación indirecta**, calculando la directa mediante ray tracing.  
- Soportar escenas con múltiples emisores, renderizado interactivo y visualización en paralelo de componentes directa/indirecta.  

## Objetivo del proyecto

El propósito es investigar si las redes neuronales pueden reemplazar parcialmente a los métodos clásicos de radiosidad, **reduciendo memoria y tiempo computacional** sin sacrificar fidelidad visual.  

Se compararon variantes del modelo, evaluando su desempeño en términos de **métricas objetivas (RMSE, SSIM)** y de apreciación visual.  

## Entorno y dependencias

El entorno se basa en las mismas tecnologías del repositorio original:  
- [Mitsuba 3](https://mitsuba-renderer.org/)  
- [Dr.Jit](https://github.com/mitsuba-renderer/drjit)  
- [PyTorch](https://pytorch.org/)  

Requisitos básicos:  
```bash
CUDA >= 11.7
Python >= 3.9
```

Instalación de dependencias principales:  
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r init/requirements.txt
```

## Ejemplos de uso

### Entrenamiento (Cornell Box)

Ejecutar `watchdog.py` con la configuración de entrenamiento:  
```bash
python watchdog.py   out_root=output/nerad   saving=[latest]   batch_size=32768   learning_rate=0.0005   rendering.spp=64   validation.image.step_size=250   validation.image.first_step=true   saving.latest.step_size=1000   n_steps=30000   lr_decay_start=10000   lr_decay_rate=0.35   lr_decay_steps=10000   lr_decay_min_rate=0.01   dataset.scene=data/NeRad_paper_scenes/cornell-box/scene.xml   rendering.config.use_autocast_rhs=false   name=cbox
```

### Renderizado interactivo

Ejecutar `test_interactive.py` sobre un experimento ya entrenado:  
```bash
python test_interactive.py   test_rendering.image.spp=64   test_rendering.image.spp_network=1   test_rendering.image.width=512   test_rendering.image.point_direct_light=false   test_rendering.image.weighted_sampling=false   test_rendering.image.paralell_rendering=true   test_rendering.image.only_indirect=true   blocksize=512   experiment=output/nerad/2025-06-26-19-07-21-cbox_indirect_scd
```

Esto abre un visualizador interactivo donde se puede modificar la posición de la cámara y la configuración de los emisores.

## Créditos

- **Repositorio original**: [Neural Radiosity](https://github.com/saeedhd96/neural-radiosity) (Hadadan et al., 2021).  
- **Implementación y extensiones**: Diego Braga – FIUBA, 2025.  

## Licencia

El código de este repositorio se publica con fines académicos.  
Consultar la licencia original de *Neural Radiosity* para más detalles.  
