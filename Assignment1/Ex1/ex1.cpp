//All code for Exercise one should go in this file!

#include "vectorbase.h"
#include "grid.h"
#include "kernel.h"

namespace Manta {

	KERNEL()
	void negateDensityKN(Grid<Real>& density, Grid<Real>& negDensity)
	{
		//TODO: fill “negDensity” with the negated values of “density”
		//Hint: The KERNEL keyword provides you with i,j,k as indices
		negDensity(i,j,k)= -1*Vec3(1,1,1)
	}

	PYTHON() void negateDensity(Grid<Real>& density, Grid<Real>& negDensity)
	{
		//TODO: Call the Kernel method
		negateDensityKN(Grid<Real>& density, Grid<Real>& negDensity)
	}
}
