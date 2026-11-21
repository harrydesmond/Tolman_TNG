r"""Reproduction of Li 2023 (ApJL 950 L14) and TNG analogue.

Li 2023 tests the distance duality relation D_L = (1+z)^2 D_A using
ultracompact radio sources from Jackson & Jannetta (2006; JJ2.29) and
Gurvits, Kellermann & Frey (1999; GKF5.0).  Under the assumption that the
mean luminosity density rho_L is independent of redshift, Li infers that
the data are fully consistent with a non-expanding (Euclidean) DDR, i.e.
D_L = (1+z) D_A, and that rescuing FRW expansion requires
rho_L \propto (1+z)^3.

This subpackage reproduces Li's procedure on his radio data and applies the
same procedure to TNG mock galaxies from the Tolman pipeline, to test whether
TNG predicts the luminosity-density evolution that Li calls a "conspiracy".
Results are written to results/li_ddr/ and Plots/li_ddr/.
"""

from .io import load_jj229, load_gkf50, load_tng_galaxy_sample  # noqa: F401
from .li_reproduction import (  # noqa: F401
    luminosity_at_band,
    fit_size_redshift_relation,
    calibrated_angular_size,
    fit_d0_expanding,
    fit_d0_nonexpanding,
    distance_duality_ratio,
    COSMO_LI,
)
