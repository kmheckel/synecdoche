import jax
import jax.tree_util as tree
import jax.numpy as jnp
import haiku as hk


def param_count(hypernetwork_params):
    """ Count the number of learnable params in a network"""
    return sum(tree.tree_leaves(tree.tree_map(jnp.size, hypernetwork_params)))

def compression_ratio(hypernet_params, target_params):
    """ Compute the savings from using a hypernetwork based approach"""
    return param_count(hypernet_params) / param_count(target_params)


class RandomProjection(hk.Module):
    """
    Use random projection to translate a set of optimizable/evovable weights into
    for each layer.
    """

    def __init__(self, key, embedding_dim, network_params, rademacher=False):
        super().__init__()
        # PyTree data needed to reconstruct net
        self.tgt_treedef = tree.tree_structure(network_params)
        self.tgt_sizes = tree.tree_map(jnp.size, network_params)
        self.num_tgt_layers = len(tree.tree_leaves(network_params))
        self.target_layer_shapes = tree.tree_map(jnp.shape, network_params)
        
        # hypernetwork dimensions
        self.embedding_dim = embedding_dim
        self.hyper_out_dim = tree.tree_reduce(max, self.tgt_sizes)
        
        if not rademacher:
            self.projection_matrix = jax.random.normal(key, 
                                                       [self.embedding_dim, self.hyper_out_dim])
        else:
            self.projection_matrix = jax.random.rademacher(key, 
                                                          [self.embedding_dim, self.hyper_out_dim])
        
    def __call__(self):
        embeddings = hk.get_parameter("w", [self.num_tgt_layers, self.embedding_dim],
                                      init=hk.initializers.TruncatedNormal())
        projections = embeddings @ self.projection_matrix
        # tgt_layers X max_layer_size matrix
        
        layer_projections = jnp.split(projections, self.num_tgt_layers)
                
        rebuilt_tree = tree.tree_unflatten(self.tgt_treedef, layer_projections)
        resized_tree = tree.tree_map(lambda layer, size: layer[1,:size], 
                                     rebuilt_tree, 
                                     self.tgt_sizes
                                    )
        net = tree.tree_map(jnp.reshape, resized_tree, self.target_layer_shapes)
        return net

class DCT(hk.Module):
    """
    Discrete Cosine Transform powered "hypernet" inspired by Compressed Weight Search.

    Jan Koutnik, Faustino Gomez, and Jürgen Schmidhuber. 2010. 
    Evolving neural networks in compressed weight space. 
    In Proceedings of the 12th annual conference on Genetic and evolutionary computation (GECCO '10). 
    Association for Computing Machinery, New York, NY, USA, 619–626. 
    https://doi.org/10.1145/1830483.1830596
    """

    def __init__(self, embedding_dim, network_params):
        super().__init__()
        # PyTree data needed to reconstruct net
        self.tgt_treedef = tree.tree_structure(network_params)
        self.tgt_sizes = tree.tree_map(jnp.size, network_params)
        self.num_tgt_layers = len(tree.tree_leaves(network_params))
        self.target_layer_shapes = tree.tree_map(jnp.shape, network_params)
        
        # hypernetwork dimensions
        self.embedding_dim = embedding_dim
        self.hyper_out_dim = tree.tree_reduce(max, self.tgt_sizes)
        
    def __call__(self):
        embeddings = hk.get_parameter("w", [self.num_tgt_layers, self.embedding_dim],
                                      init=hk.initializers.TruncatedNormal())
        
        projections = jax.scipy.fft.idctn(embeddings, s=[self.num_tgt_layers, self.hyper_out_dim])
        # tgt_layers X max_layer_size matrix
        
        layer_projections = jnp.split(projections, self.num_tgt_layers)
                
        rebuilt_tree = tree.tree_unflatten(self.tgt_treedef, layer_projections)
        resized_tree = tree.tree_map(lambda layer, size: layer[1,:size], 
                                     rebuilt_tree, 
                                     self.tgt_sizes
                                    )
        net = tree.tree_map(jnp.reshape, resized_tree, self.target_layer_shapes)
        return net


class MatrixFactorization(hk.Module):
    """
    Attempt to learn network weights through matrix factorization. 
    In its current implementation its only likely efficient if 
    the number of network layers >> rank of the model.

    projections = [target_layers x rank] @ [rank x max_target_layer_size]
    """

    def __init__(self, rank, network_params):
        super().__init__()
        # PyTree data needed to reconstruct net
        self.tgt_treedef = tree.tree_structure(network_params)
        self.tgt_sizes = tree.tree_map(jnp.size, network_params)
        self.num_tgt_layers = len(tree.tree_leaves(network_params))
        self.target_layer_shapes = tree.tree_map(jnp.shape, network_params)
        
        # hypernetwork dimensions
        self.rank = rank
        self.hyper_out_dim = tree.tree_reduce(max, self.tgt_sizes)
        
    # more efficient if num_tgt_layers >> self.rank...
    def __call__(self):
        left_embeddings = hk.get_parameter("w", [self.num_tgt_layers, self.rank],
                                      init=hk.initializers.TruncatedNormal())
        
        right_embeddings = hk.get_parameter("b", [self.rank, self.hyper_out_dim],
                                      init=hk.initializers.TruncatedNormal())
        
        projections = left_embeddings @ right_embeddings
        # tgt_layers X max_layer_size matrix
        
        layer_projections = jnp.split(projections, self.num_tgt_layers)
                
        rebuilt_tree = tree.tree_unflatten(self.tgt_treedef, layer_projections)
        resized_tree = tree.tree_map(lambda layer, size: layer[1,:size], 
                                     rebuilt_tree, 
                                     self.tgt_sizes
                                    )
        net = tree.tree_map(jnp.reshape, resized_tree, self.target_layer_shapes)
        return net

class Hypernetwork(hk.Module):
    """
    Static Hypernetwork implementation inspired by Ha et. al. 2016 and 
    the implementation in evosax. Like the other implementations in this
    mini-library, the projections are fit to the max layer size
    and then downselected to fit the shaping. This is less efficient than 
    projecting to a moderately sized dimension and then replicating the output.




    “Hypernetworks.” Hypernetworks in the Science of Complex Systems, Dec. 2013, pp. 151–76. 
    Crossref, https://doi.org/10.1142/9781860949739_0006.

    @misc{lange2022evosax,
    title={evosax: JAX-based Evolution Strategies},
    author={Robert Tjarko Lange},
    year={2022},
    eprint={2212.04180},
    archivePrefix={arXiv},
    primaryClass={cs.NE}
    }

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
        
    def __call__(self):
        embeddings = hk.get_parameter("w", [self.num_tgt_layers, self.embedding_dim],
                                      init=hk.initializers.TruncatedNormal())
        projections = hk.nets.MLP([self.latent_dim, self.latent_dim])(embeddings)
        # tgt_layers X max_layer_size matrix
        
        layer_projections = jnp.split(projections, self.num_tgt_layers)
                
        rebuilt_tree = tree.tree_unflatten(self.tgt_treedef, layer_projections)
        # this is ugly... sorry...
        resized_tree = tree.tree_map(lambda layer, size: jnp.pad(layer[1,:size], 
                                                                 (0,max(0,size-layer.size)), 
                                                                 mode="wrap"), 
                                     rebuilt_tree, 
                                     self.tgt_sizes
                                    )
        net = tree.tree_map(jnp.reshape, resized_tree, self.target_layer_shapes)
        return net