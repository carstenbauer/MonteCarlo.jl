"""
    hopping_matrix(mc::DQMC, m::Model)

Calculates the hopping matrix \$T_{i\\sigma, j\\sigma '}\$ where \$i, j\$ are 
site indices and \$\\sigma , \\sigma '\$ are flavor indices (e.g. spin indices). 
The hopping matrix should also contain potential chemical potential terms on the 
diagonal.

A matrix element is the hopping amplitude for a hopping process: \$j,\\sigma ' 
\\rightarrow i,\\sigma\$.

Regarding the order of indices, if `T[i, σ, j, σ']` is your desired 4D hopping 
array, then `reshape(T, (n_sites * n_flavors, :))` is the hopping matrix.
"""
hopping_matrix(mc::DQMC, m::Model) = throw(MethodError(hopping_matrix, (mc, m)))