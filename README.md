# pyOriMap

## Description
Numpy implementation of Hamilton's "Explicit Equations for the Stresses beneath a Sliding Spherical Contact", 1983 (https://doi.org/10.1243/PIME_PROC_1983_197_076_02).

## Authors and acknowledgment
Sylvain Dancette (sylvain.dancette@insa-lyon.fr).

## License
MIT license.

## Installation
Requires numpy and pyvista.

```
pip install pymilton # to be done
```

## Usage
From scratch, starting by defining a mesh grid and a few contact parameters:
```
from pymilton import hamilton as hami

params = hami.HamiltonParameters()
params.dimensions = (52,52,26)
params.spacing = (0.1, 0.1, 0.1)
params.origin = (-2.6, -2.6, 0.)
params.E = 1.55
params.nu = 0.49
params.R = 9.42
params.P = 1.48

params.get_Hertzian_params()

mesh = hami.Hamilton(params=params)
```

Alternatively, starting from an existing mesh in Legacy VTK file format:
```
mesh = hami.read_from_vtk("mesh.vtk")
mesh.params.E = 1.55
mesh.params.nu = 0.49
mesh.params.R = 9.42
mesh.params.P = 1.48

mesh.compute_stress_explicit()
```

## Support
Open issue or email to sylvain.dancette@insa-lyon.fr

## Roadmap

## Contributing
Open to contributions, contact sylvain.dancette@insa-lyon.fr.
