import pytest

import susiepca as sp


n_dim = 20
p_dim = 100
l_dim = 10
z_dim = 4


def test_simulation():
    with pytest.raises(ValueError):
        # seed is not interger
        sp.sim.generate_sim(
            seed=1.1, l_dim=l_dim, n_dim=n_dim, p_dim=p_dim, z_dim=z_dim
        )
        # l_dim>p_dim
        sp.sim.generate_sim(seed=1, l_dim=110, n_dim=n_dim, p_dim=p_dim, z_dim=z_dim)
        # l_dim > p_dim/z_dim
        sp.sim.generate_sim(seed=1, l_dim=30, n_dim=n_dim, p_dim=p_dim, z_dim=z_dim)
        # l_dim<0
        sp.sim.generate_sim(seed=1, l_dim=-1, n_dim=n_dim, p_dim=p_dim, z_dim=z_dim)

        # z_dim>p_dim
        sp.sim.generate_sim(seed=1, l_dim=l_dim, n_dim=n_dim, p_dim=p_dim, z_dim=110)
        # z_dim>n_dim
        sp.sim.generate_sim(seed=1, l_dim=l_dim, n_dim=n_dim, p_dim=p_dim, z_dim=30)

        # z_dim<=0
        sp.sim.generate_sim(seed=1, l_dim=l_dim, n_dim=n_dim, p_dim=p_dim, z_dim=-1)
        #
        sp.sim.generate_sim(
            seed=1, l_dim=l_dim, n_dim=n_dim, p_dim=p_dim, z_dim=z_dim, effect_size=0
        )
