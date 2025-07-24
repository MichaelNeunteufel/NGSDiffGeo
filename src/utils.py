import ngsolve

__all__ = [ "Sphere2", "Sphere3", "PoincareDisk", "HyperbolicH2", "HyperbolicH3", "Heisenberg", "CigarSoliton", "WarpedProduct" ]

# Standard metric on S^2. x and y are interpreted as angles; x in [0,pi], y in [0,2*pi). Has constant positive curvature.
Sphere2 = ngsolve.CF( (1,0,
                       0,ngsolve.sin(ngsolve.x)**2), dims=(2,2) )

# Standard metric on S^3. x, y, and z are interpreted as angles; x in [0,pi], y in [0,pi], z in [0,2*pi). Has constant positive curvature.
Sphere3 = ngsolve.CF( (1,0,0, 
                       0,ngsolve.sin(ngsolve.x)**2,0,
                       0,0,ngsolve.sin(ngsolve.x)**2*ngsolve.sin(ngsolve.y)**2), dims=(3,3) )

# Hyperbolic metric on the Poincare Disk B_1(0)= {(x,y) in R^2 : x^2+y^2 < 1}. Has constant negative curvature.
PoincareDisk = 4/(1-ngsolve.x**2-ngsolve.y**2)**2 * ngsolve.Id(2)


# Hyperbolic metric on H2={(x,y) in R^2 : y > 0}. Has constant negative curvature.
HyperbolicH2 = 1/ngsolve.y**2 * ngsolve.Id(2)

# Hyperbolic metric on H3={(x,y,z) in R^3 : z > 0}. Has constant negative curvature.
HyperbolicH3 = 1/ngsolve.z**2 * ngsolve.Id(3)

# Heisenberg metric on R^3. Has non-zero Ricci curvature
Heisenberg  = ngsolve.CF( (1,0,0,
                           0,1+ngsolve.x**2,-ngsolve.x,
                           0,-ngsolve.x,1), dims=(3,3) )

# Cigar Soliton Metric on R^2
CigarSoliton = 1/(1+ngsolve.x**2+ngsolve.y**2)*ngsolve.Id(2)

# Warped Product Metric on R^3
WarpedProduct = ngsolve.CF( (ngsolve.exp(2*ngsolve.z),0,0, 
                             0,ngsolve.exp(2*ngsolve.z),0,
                             0,0,1), dims=(3,3) )