# test_tensor_core.py
import drjit as dr
import drjit.nn as nn
from drjit.cuda import Float16, TensorXf16

# Crear pesos y bias
A = dr.normal(TensorXf16, (16, 16))
b = dr.normal(TensorXf16, (16,))

# Empaquetar
buffer, A_view, b_view = nn.pack(A, b)

# Crear vector de entrada como CoopVec (requerido para usar Tensor Cores)
x = nn.CoopVec(*dr.rand(Float16, 16))

# Ejecutar producto (usa Tensor Cores si están habilitados)
y = nn.matvec(A_view, x, b_view)

# Forzar evaluación
print("Resultado:", y)
