#
# Simple example scene for a 2D simulation
# Simulation of a buoyant smoke density plume
#

# Setup simulation
from manta import *

# solver params
res = 128                                                                   # (1)
gs  = vec3(res,2*res,1)                                                         # (1)
s          = Solver(name='main', gridSize = gs, dim=2)                          # (1)
s.timestep = 1.0                                                                # (2)
timings    = Timings()                                                          # for output

# prepare grids
flags    = s.create(FlagGrid)                                                   # (3)
vel      = s.create(MACGrid)                                                    # (3)
density  = s.create(RealGrid)                                                   # (3)
pressure = s.create(RealGrid)                                                   # (3)

flags.initDomain()                                                              # (4)
flags.fillGrid()                                                                # (4)

if (GUI):                                                                       # for output
	gui = Gui()                                                                 # for output
	gui.show( True )                                                            # for output
	gui.pause()                                                                 # for output

source = s.create(Cylinder, center=gs*vec3(0.5,0.05,0.5)                        # (5)
                          , radius=res*0.075                                    # (5)
                          , z=gs*vec3(0, 0.01, 0))                              # (5)
	
# time loop
for t in range(1000):

	source.applyToGrid(grid=density, value=1.)                                  # (i)

	advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2)             # (ii)
	advectSemiLagrange(flags=flags, vel=vel, grid=vel,     order=2)             # (ii)
	
	setWallBcs(flags=flags, vel=vel)                                            # (iii)
	addBuoyancy(density=density, vel=vel, gravity=vec3(0,-5e-5,0), flags=flags) # (iv)
	
	solvePressureGS(flags=flags, vel=vel, pressure=pressure, gsAccuracy=1e-4)   # (v)
	#print(getMaxDivergence(vel,flags)) # for debugging

	timings.display()                                                           # for output
	s.step()                                                                    # (vi)

	if t == 250:
		gui.screenshot( 'shot_%04d.png' % t);
	if t == 500:
		gui.screenshot( 'shot_%04d.png' % t);
	if t == 750:
		gui.screenshot( 'shot_%04d.png' % t);