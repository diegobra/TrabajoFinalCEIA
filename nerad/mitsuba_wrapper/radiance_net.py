from typing import Any

import drjit as dr
import mitsuba as mi
import torch
import torch.nn as nn

from nerad.mitsuba_wrapper import MitsubaWrapper, wrapper_registry
from nerad.model.tcnn_embedding import TcnnEmbedding
from nerad.model.multires_grid import MutliResGrid
from nerad.utils.mitsuba_utils import vec_to_tens_safe


def create_embedding(config):
    if config['otype'] == 'SparseGrid':
        embedding = MutliResGrid(**config)
    elif config['otype'] == 'DenseGrid':
        embedding = MutliResGrid(**config)
    else:
        embedding = TcnnEmbedding(config)
    return embedding

def embed(input_, embedding):
    embed_type = embedding.embedding_type
    net_in = None
    match embed_type:
        case "identity":
            net_in = input_
        case "SparseGrid":
            net_in = torch.cat([input_, embedding(input_)], dim=-1)
        case "DenseGrid":
            net_in = torch.cat([input_, embedding(input_)], dim=-1)
        case "HashGrid":
            net_in = torch.cat([input_, embedding(input_)], dim=-1)
        case "Grid":
            net_in = torch.cat([input_, embedding((input_-0.5))], dim=-1)
        case "Frequency":
            net_in = embedding(2*input_-1)
        case "SphericalHarmonics":
            net_in = embedding(input_)
        case _:
            raise Exception("Unhandled embedding")

    return net_in


class RadianceMLP(nn.Module):
    def __init__(
        self,
        width: int,
        hidden: int,
        position_embedding: dict[str, Any],
        direction_embedding: dict[str, Any],
        emitter_position_embedding: dict[str, Any],
        scene_properties_input: bool,
        emitter_input: bool
    ):
        super().__init__()
        self.scene_properties_input = scene_properties_input
        self.pos_emb = create_embedding(position_embedding)
        self.dir_emb = create_embedding(direction_embedding)
        self.emitter_pos_emb = create_embedding(emitter_position_embedding)
        self.emitter_input = emitter_input

        def embed_size(in_vector, embedding):
            return embed(torch.zeros(1, in_vector).cuda(), embedding).shape[-1]

        #input size : points + direction
        in_size = embed_size(3, self.pos_emb) + embed_size(3, self.dir_emb)

        if scene_properties_input:
            in_size += embed_size(3, self.dir_emb)      #normal
            in_size += 3                                #albedo

        if emitter_input:
            in_size += embed_size(3, self.emitter_pos_emb) + 3 + 1 + 3 # posición, normal y radio del emisor de entrada
            #in_size += 3 + 3 + 1 + 3 # posición, normal, radio y radiancia del emisor de entrada

        hidden_layers = []
        for _ in range(hidden):
            hidden_layers.append(nn.Linear(width, width))
            hidden_layers.append(nn.ReLU(inplace=True))

        self.network = nn.Sequential(
            nn.Linear(in_size, width),
            nn.ReLU(inplace=True),
            *hidden_layers,
            nn.Linear(width, 3),
        )

    def forward(self, points, dirs, normals, albedo, emitter_pos=None, emitter_normal=None, emitter_radius=None, emitter_radiance=None):
        net_in = torch.cat(
            [
                embed(points, self.pos_emb),    # (HashGrid) torch.Size([32768, 99])
                embed(dirs, self.dir_emb)       # (Identity) torch.Size([32768, 3])
            ],
            dim=-1,
        )

        # A este punto net_in.shape = torch.Size([32768, 102])

        if self.scene_properties_input:
            net_in = torch.cat(
                [
                    net_in,                         # torch.Size([32768, 102])
                    embed(normals, self.dir_emb),   # torch.Size([32768, 3])
                    albedo],                        # torch.Size([32768, 3])
                dim=-1,
            )
        # Sizes of tensors must match except in dimension 0. Expected size 108 but got size 3 for tensor number 1 in the list.
        if self.emitter_input:
            net_in = torch.cat(
                [
                    net_in,                             # torch.Size([32768, 108])
                    embed(emitter_pos, self.emitter_pos_emb),   # torch.Size([32768, 99])
                    #emitter_pos,   # torch.Size([32768, 99])
                    emitter_normal,                     # torch.Size([32768, 3])
                    emitter_radius,
                    emitter_radiance],                    # torch.Size([32768, 1])
                dim=-1,
            )

        # A este punto net_in.shape = torch.Size([32768, 108])

        # A la red se le va a pasar:
        #   points              (32768, 3),
        #   embedding<points>   (32768, 96),
        #   dirs                (32768, 3)
        #   normals             (32768, 3),
        #   albedo              (32768, 3)

        ret = self.network(net_in) # ret.shape = torch.Size([32768, 3])
        return torch.abs(ret)

@wrapper_registry.register("radiance_net")
class MitsubaRadianceNetworkWrapper(MitsubaWrapper):
    def __init__(
        self,
        width: int,
        hidden: int,
        position_embedding: dict[str, Any],
        direction_embedding: dict[str, Any],
        emitter_position_embedding: dict[str, Any],
        scene_min: Any,
        scene_max: Any,
        scene_properties_input,
        emitter_input
    ):
        super().__init__(scene_min, scene_max, "radiance_net")
        self.network = RadianceMLP(width, hidden, position_embedding, direction_embedding, emitter_position_embedding, scene_properties_input, emitter_input)

    def _eval(self, pts, dirs, norms, albedo, emitter_pos=None, emitter_normal=None, emitter_radius=None, emitter_radiance=None):
        p_tensor = vec_to_tens_safe(pts + self.grad_activator) # TensorXf(shape=(32768, 3))
        d_tensor = vec_to_tens_safe(dirs) # TensorXf(shape=(32768, 3))
        n_tensor = vec_to_tens_safe(norms) # TensorXf(shape=(32768, 3))
        alb_tensor = vec_to_tens_safe(albedo) # TensorXf(shape=(32768, 3))
        if self.network.emitter_input:
            emitter_pos_tensor = vec_to_tens_safe(dr.tile(emitter_pos, len(pts[0])))
            emitter_normal_tenosr = vec_to_tens_safe(dr.tile(emitter_normal, len(pts[0])))
            #emitter_radius_tensor = vec_to_tens_safe(emitter_radius)
            emitter_radius_tensor = mi.TensorXf(dr.tile(emitter_radius, len(pts[0])), shape=(len(pts[0]), 1))
            emitter_radiance_tensor = vec_to_tens_safe(dr.tile(emitter_radiance, len(pts[0])))
            torch_out = self.eval_torch(
                p_tensor, d_tensor, n_tensor, alb_tensor, emitter_pos_tensor, emitter_normal_tenosr, emitter_radius_tensor, emitter_radiance_tensor)
        else:
            torch_out = self.eval_torch(
                p_tensor, d_tensor, n_tensor, alb_tensor)

        output = dr.unravel(mi.Vector3f, torch_out.array)
        return dr.abs(output)

    @dr.wrap_ad(source='drjit', target='torch')
    def eval_torch(self, pts, dirs, norms, albedo, emitter_pos=None, emitter_normal=None, emitter_radius=None, emitter_radiance=None):
        return self.network(pts, dirs, norms, albedo, emitter_pos, emitter_normal, emitter_radius, emitter_radiance)

    def _traverse(self, callback):
        callback.put_parameter("network", self.network, mi.ParamFlags.Differentiable)
