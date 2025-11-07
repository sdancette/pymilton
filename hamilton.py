# -*- coding: utf-8 -*-

# pymilton/hamilton.py

"""
Numpy implementation of Hamilton's "Explicit Equations for the Stresses beneath a Sliding Spherical Contact", 1983.

The module contains the following functions:

- `xxx(n)` - generate xxx
"""

import logging
import numpy as np
import pyvista as pv

from dataclasses import dataclass, field
from typing import List, Tuple

#logging.basicConfig(filename='hamilton.log', level=logging.INFO, format='%(asctime)s %(message)s')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DTYPEf = np.float64
DTYPEi = np.int64

_EPS = 1e-9

@dataclass
class HamiltonParameters:
    filename: str = 'thegrid.vtk'
    dimensions: Tuple[int, int, int] = (64, 64, 32)
    spacing: Tuple[float, float, float] = (1., 1., 1.)
    origin: Tuple[float, float, float] = (-32., -32., 0.)
    dim3D: bool = True
    R: float = 64.
    P: float = 1.0
    Q: float = 0.0
    E: float = 1.0
    nu: float = 0.3

    def get_Hertzian_params(self):
        """
        Compute contact radius and maximum contact pressure.
        """
        self.lame = self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        self.G = self.E/(2*(1+self.nu))
        self.B = self.E/(3*(1-2*self.nu))

        self.K = 4/3*self.E/(1-self.nu**2)
        self.a = (self.P*self.R/self.K)**(1./3) # contact radius
        self.p0 = 3*self.P/2/np.pi/self.a**2 # contact pressure
        

class Hamilton(pv.UnstructuredGrid):
    """
    Hamilton class (inheriting from pyvista UnstructuredGrid).

    Data is stored at the vertices (as point_data) by default rather than at the cell centers.

    Parameters
    ----------
    uinput : str, vtk.UnstructuredGrid, pyvista.UnstructuredGrid, optional
        Filename or dataset to initialize the unstructured grid from.

    Attributes
    ----------
    filename : str
        String corresponding to the filename.

    Methods
    -------
    compute_stress()
        Compute the stress field from Hamilton's explicit equations.
    """

    def __init__(self, uinput=None, params=HamiltonParameters()):
        self.params = params
        if uinput is None:
            logging.info("Initializing grid from params.")
            meshbox = pv.ImageData()
            meshbox.dimensions = np.array(params.dimensions) + 1
            meshbox.origin = params.origin
            meshbox.spacing = params.spacing
            meshbox = meshbox.cast_to_structured_grid()
            super().__init__(meshbox) #meshbox.cast_to_unstructured_grid()            
        else:
            logging.info("Initializing grid from {}.".format(uinput))
            super().__init__(uinput)

            logging.info("Updating parameters based on vtk file properties.")
            elsize = (self.volume / self.n_cells)**(1./3)
            self.params.dimensions =   ((self.bounds[1]-self.bounds[0])/elsize, 
                                        (self.bounds[3]-self.bounds[2])/elsize, 
                                        (self.bounds[5]-self.bounds[4])/elsize)
            self.params.spacing = (elsize, elsize, elsize)
            self.params.origin = (self.bounds[0], self.bounds[2], self.bounds[4])

    def compute_stress_explicit(self):
        """
        Compute Hamilton's explicit stress components.
        """
        logging.info("Starting to compute stress field.")
        x = self.points[:,0]
        y = self.points[:,1]
        z = self.points[:,2]
        r = np.sqrt(x**2 + y**2)
        
        self.params.get_Hertzian_params()
        a = self.params.a
        P = self.params.P
        nu = self.params.nu

        self.point_data['Sij'] =      np.zeros((self.n_points,9), dtype=DTYPEf)
        self.point_data['Mises'] =    np.zeros(self.n_points, dtype=DTYPEf)
        self.point_data['Pressure'] = np.zeros(self.n_points, dtype=DTYPEf)

        A = r**2 + z**2 - a**2
        S = np.sqrt(A**2 + 4 * a**2 * z**2)
        M = np.sqrt((S + A)/2)
        N = np.sqrt((S - A)/2)
        phi = np.arctan(a/M)
        G = M**2 - N**2 + z*M - a*N
        H = 2*M*N + a*M + z*N

        whr0 = (r < _EPS)
        self['Sij'][~whr0,0] = 3*P/(2*np.pi*a**3)*( (1+nu)*z[~whr0]*phi[~whr0] + \
                       1/(r[~whr0]**2) * ((y[~whr0]**2 - x[~whr0]**2)/(r[~whr0]**2)*((1-nu)*N[~whr0]*z[~whr0]**2 - 
                       (1-2*nu)/3*(N[~whr0]*S[~whr0] + 2*A[~whr0]*N[~whr0] + a**3)-nu*M[~whr0]*z[~whr0]*a) \
                       -N[~whr0]*(x[~whr0]**2 + 2*nu*y[~whr0]**2) - M[~whr0]*x[~whr0]**2*z[~whr0]*a/S[~whr0]) )
        
        self['Sij'][~whr0,4] = 3*P/(2*np.pi*a**3)*((1+nu)*z[~whr0]*phi[~whr0] + \
                       1/(r[~whr0]**2)*((x[~whr0]**2 - y[~whr0]**2)/(r[~whr0]**2)*((1-nu)*N[~whr0]*z[~whr0]**2 - (1-2*nu)/3*(N[~whr0]*S[~whr0] + 
                        2*A[~whr0]*N[~whr0] + a**3)-nu*M[~whr0]*z[~whr0]*a) \
                       -N[~whr0]*(y[~whr0]**2 + 2*nu*x[~whr0]**2) - M[~whr0]*y[~whr0]**2*z[~whr0]*a/S[~whr0]))
        
        self['Sij'][~whr0,8] = 3*P/(2*np.pi*a**3)*(-N[~whr0] + a*z[~whr0]*M[~whr0]/S[~whr0])
        
        self['Sij'][~whr0,1] = 3*P/(2*np.pi*a**3)*(x[~whr0]*y[~whr0]*(1-2*nu)/r[~whr0]**4*(-N[~whr0]*r[~whr0]**2 + 2/3*N[~whr0]*(S[~whr0] + 
                          2*A[~whr0])-z[~whr0]*(z[~whr0]*N[~whr0] + a*M[~whr0]) + 2/3*a**3) + \
                        x[~whr0]*y[~whr0]*z[~whr0]/r[~whr0]**4*(-a*M[~whr0]*r[~whr0]**2/S[~whr0] - z[~whr0]*N[~whr0] + a*M[~whr0]))
        self['Sij'][~whr0,3] = self['Sij'][~whr0,1]
        
        self['Sij'][~whr0,5] = 3*P/(2*np.pi*a**3)*(-z[~whr0]*(y[~whr0]*N[~whr0]/S[~whr0] - y[~whr0]*z[~whr0]*H[~whr0]/(G[~whr0]**2+H[~whr0]**2)))
        self['Sij'][~whr0,7] = self['Sij'][~whr0,5]
        
        self['Sij'][~whr0,2] = 3*P/(2*np.pi*a**3)*(-z[~whr0]*(x[~whr0]*N[~whr0]/S[~whr0] - x[~whr0]*z[~whr0]*H[~whr0]/(G[~whr0]**2+H[~whr0]**2)))
        self['Sij'][~whr0,6] = self['Sij'][~whr0,2]
        
        self['Sij'][whr0,0] = 3*P/(2*np.pi*a**3)*( (1+nu)*(z[whr0]*np.arctan(a/z[whr0]) - a) + a**3/(2*(a**2 + z[whr0]**2)) )
        self['Sij'][whr0,4] = self['Sij'][whr0,0]
        self['Sij'][whr0,8] = 3*P/(2*np.pi*a**3)*(-a**3/(a**2+z[whr0]**2))

        self['Mises'] = np.sqrt(2)/2*np.sqrt((self['Sij'][:,0] - self['Sij'][:,4])**2 + 
                                     (self['Sij'][:,4] - self['Sij'][:,8])**2 + 
                                     (self['Sij'][:,8] - self['Sij'][:,0])**2 + 
                                     6*(self['Sij'][:,1]**2 + self['Sij'][:,2]**2 + self['Sij'][:,5]**2)) 
        self['Pressure'] = -(self['Sij'][:,0] + self['Sij'][:,4] + self['Sij'][:,8])/3
        
        logging.info("Finished to compute stress field.")
        
        self.save(self.params.filename[:-4]+'.vtk')
        
    def compute_strain_explicit(self):
        """
        Compute Hamilton's strain components based on explicit stress.
        """
        logging.info("Starting to compute strain field.")

        self.point_data['Eij'] =    np.zeros((self.n_points,9), dtype=DTYPEf)
        self.point_data['Eequiv'] = np.zeros(self.n_points, dtype=DTYPEf)
        self.point_data['Ehydro'] = np.zeros(self.n_points, dtype=DTYPEf)
    
    def compute_divergence_stress(self):
        """
        Compute divergence of stress tensor.
        """
        logging.info("Starting to compute divergence of stress.")

        self.point_data['Saxis1'] = np.zeros((self.n_points, 3), dtype=DTYPEf)
        self.point_data['Saxis2'] = np.zeros((self.n_points, 3), dtype=DTYPEf)
        self.point_data['Saxis3'] = np.zeros((self.n_points, 3), dtype=DTYPEf)
        
        self.point_data['divSijAx1'] = np.zeros(self.n_points, dtype=DTYPEf)
        self.point_data['divSijAx2'] = np.zeros(self.n_points, dtype=DTYPEf)
        self.point_data['divSijAx3'] = np.zeros(self.n_points, dtype=DTYPEf)

    def compute_displacement(self):
        """
        Compute displacement by numerical integration of harmonic stress function Psi and Omega.
        """
        logging.info("Starting to compute displacement.")

        self.point_data['Psi'] =   np.zeros(self.n_points, dtype=DTYPEf)
        self.point_data['Omega'] = np.zeros(self.n_points, dtype=DTYPEf)
        self.point_data['U'] =     np.zeros((self.n_points, 3), dtype=DTYPEf)
        
def read_from_vtk(filename):
    """
    Read a .vtk file and return an Hamilton object.
    """
    logging.info("Reading data from {}.".format(filename))

    params = HamiltonParameters(filename=filename)

    hamigrid = Hamilton(uinput=filename, params=params)
 
    logging.info("Finished reading from {}.".format(filename))

    return hamigrid
