#
# Simple example scene for a 2D simulation
# Simulation of a buoyant smoke density plume
#
from manta import *

# solver params
res = 64
gs = vec3(res,res,1)
s = Solver(name='main', gridSize = gs, dim=2)
s.timestep = 0.25

# prepare grids
flags = s.create(FlagGrid)

flags.initDomain()
flags.fillGrid()

temperature = s.create(RealGrid)
alpha = 0.5

if (GUI):
	gui = Gui()
	gui.show( True ) 
	gui.pause()
	
source = s.create(Cylinder, center=gs*vec3(0.5,0.5,0.5), radius=res*0.2, z=gs*vec3(0, 0.2, 0))
source.applyToGrid(grid=temperature, value=1)
	
#main loop
for t in range(800):
	diffuseTemperatureExplicit(flags,temperature,alpha)
	#diffuseTemperatureImplicit(temperature, ...)
	s.step()
	if (t == 200):
		gui.screenshot('diffuseBox_%04d.png' % t) 
	if(t%100 == 0):
		gui.pause()