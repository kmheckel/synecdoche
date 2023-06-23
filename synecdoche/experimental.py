import jax
import jax.tree_util as tree
import jax.numpy as jnp
import haiku as hk

class DynamicHypernetwork(hk.Module):
    def __init__(self, embedding_dim, latent_dim, network_params):
        super().__init__()
        # PyTree data needed to reconstruct net
        self.tgt_treedef = tree.tree_structure(network_params)
        self.tgt_sizes = tree.tree_map(jnp.size, network_params)
        self.num_tgt_layers = len(tree.tree_leaves(network_params))
        self.target_layer_shapes = tree.tree_map(jnp.shape, network_params)
        
        # hypernetwork dimensions
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.hyper_out_dim = tree.tree_reduce(max, self.tgt_sizes)
        
    def __call__(self, x):
        avg = jnp.mean(x, axis=0)
        layer_inputs = jnp.repeat(jnp.expand_dims(avg, 0), self.num_tgt_layers, 0)
        projections = hk.nets.MLP([self.embedding_dim, self.latent_dim, self.hyper_out_dim])(layer_inputs)

        layer_projections = jnp.split(projections, self.num_tgt_layers)
                
        rebuilt_tree = tree.tree_unflatten(self.tgt_treedef, layer_projections)
        resized_tree = tree.tree_map(lambda layer, size: layer[1,:size], # the axis=1 is odd.
                                     rebuilt_tree, 
                                     self.tgt_sizes
                                    )
        net = tree.tree_map(jnp.reshape, resized_tree, self.target_layer_shapes)
        return net