import jax
import jax.tree_util as tree
import jax.numpy as jnp
import haiku as hk

class DynamicHypernetwork(hk.Module):
    """
    Hypernetwork that takes the current batch as inputs, averages the samples, and then uses that average to predict weights for the layer.
    """

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
        
    def __call__(self, x):
        avg = jnp.mean(x, axis=0)
        layer_inputs = jnp.repeat(jnp.expand_dims(avg, 0), self.num_tgt_layers, 0)
        projections = hk.nets.MLP([self.embedding_dim, self.latent_dim, self.latent_dim])(layer_inputs)

        layer_projections = jnp.split(projections, self.num_tgt_layers)
                
        rebuilt_tree = tree.tree_unflatten(self.tgt_treedef, layer_projections)
        resized_tree = tree.tree_map(lambda layer, size: jnp.pad(layer[1,:size], 
                                                                 (0,max(0,size-layer.size)), 
                                                                 mode="wrap"), 
                                     rebuilt_tree, 
                                     self.tgt_sizes
                                    )
        net = tree.tree_map(jnp.reshape, resized_tree, self.target_layer_shapes)
        return net