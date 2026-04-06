"""
Spectral shape matching pipeline.

Provides configuration and orchestration for mesh correspondence via
functional maps, spectral descriptors, and product manifold filtering.
"""

from collections import namedtuple

__all__ = [
    "SigConfig", "default_sig_config",
    "MatchConfig", "default_match_config",
]

SigConfig = namedtuple('SigConfig',
                       ['emin',
                        'emax',
                        'num_wks',
                        'num_hks',
                        'num_gaussian',
                        'num_signatures'])

default_sig_config = SigConfig(
                        emin=0.01,
                        emax=1000,
                        num_wks=75,
                        num_hks=25,
                        num_gaussian=18,
                        num_signatures=118)

MatchConfig = namedtuple('MatchConfig',
                            ['initial_solve_dimension',
                            'symmetry_optimisation',
                            'pmf_sigma',
                             'pmf_gamma',
                             'pmf_iters',
                             'euclidean_init'])

default_match_config = MatchConfig(
                        initial_solve_dimension=6,
                        symmetry_optimisation=True,
                        pmf_sigma=0.75,
                        pmf_gamma=0.75,
                        pmf_iters=4,
                        euclidean_init=0.5)