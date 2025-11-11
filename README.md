# pymilton

## Description
Numpy implementation of Hamilton's "Explicit Equations for the Stresses beneath a Sliding Spherical Contact", 1983 (https://doi.org/10.1243/PIME_PROC_1983_197_076_02).

## Authors and acknowledgment
Sylvain Dancette (sylvain.dancette@insa-lyon.fr).

## License
MIT license.

## Installation
Requires numpy and pyvista.

```
pip install -U numpy pyvista

pip install pymilton # to be done
```

## Usage
From scratch, starting by defining a mesh grid:
```
from pymilton import hamilton as hami

params = hami.HamiltonParameters()
params.dimensions = (52,52,26)
params.spacing = (0.1, 0.1, 0.1)
params.origin = (-2.6, -2.6, 0.)

mesh = hami.Hamilton(params=params)
```
Alternatively, starting from an existing mesh in Legacy VTK file format:
```
mesh = hami.read_from_vtk("mesh.vtk")
```
At this point the mesh is available as an Hamilton object, 
from which stress, strain and displacement field will be computable.

To do so, define a few contact parameters and the corresponding contact radius and maximum pressure:
```
mesh.params.E = 1.55  # Young's modulus
mesh.params.nu = 0.49 # Poisson's coefficient
mesh.params.R = 9.42  # indenter radius of curvature 
mesh.params.P = 1.48  # Normal load
mesh.params.get_Hertzian_params()
```
Then you can compute the stress, strain or displacement field: 
```
mesh.compute_stress_explicit()
mesh.compute_strain_from_stress()
mesh.compute_displacement()
```

## Support
Open issue or email to sylvain.dancette@insa-lyon.fr

## Roadmap

## Contributing
Open to contributions, contact sylvain.dancette@insa-lyon.fr.
