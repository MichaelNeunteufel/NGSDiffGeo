NGSDiffGeo
==========

NGSDiffGeo extends NGSolve with differential-geometry operations, with a focus
on Riemannian manifolds. Start with the geometry foundations below, then follow
the Regge and curvature tutorials or use the calculus notebooks as references.

We gratefully acknowledge support by the Austrian Science Foundation
`FWF <https://www.fwf.ac.at>`_ through
`project J4824, Numerical analysis of curvatures from Regge finite elements
<https://www.fwf.ac.at/en/research-radar/10.55776/J4824>`_.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   install
   license

.. toctree::
   :caption: Geometry foundations
   :maxdepth: 1

   tutorials/01_riemannian_manifolds.ipynb
   tutorials/02_covariant_derivatives.ipynb
   tutorials/03_curvatures.ipynb
   tutorials/04_gauss_bonnet.ipynb

.. toctree::
   :caption: Regge metrics and Gauss curvature
   :maxdepth: 1

   tutorials/05_regge_metric.ipynb
   tutorials/06_distributional_gauss_curvature.ipynb
   tutorials/07_distributional_gauss_curvature_analysis.ipynb
   tutorials/08_distributional_gauss_curvature_surface.ipynb

.. toctree::
   :caption: Higher-dimensional curvature and linearization
   :maxdepth: 1

   tutorials/09_distributional_scalar_curvature.ipynb
   tutorials/10_distributional_einstein_tensor.ipynb
   tutorials/11_distributional_riemann_curvature_tensor.ipynb
   tutorials/12_linearization_curvature_quantities.ipynb

.. toctree::
   :caption: Exterior calculus and hypersurfaces
   :maxdepth: 1

   tutorials/13_kforms_exterior_derivative.ipynb
   tutorials/14_double_forms.ipynb
   tutorials/15_double_forms_covariant_derivatives.ipynb
   tutorials/16_submanifold_geometry.ipynb

.. toctree::
   :caption: Developer Guide
   :maxdepth: 1

   developer_docs/index
