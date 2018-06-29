#include "kernel.h"
#include "grid.h"

using namespace std;
namespace Manta {

	KERNEL(bnd = 1)
	void ComputeDiv(FlagGrid& flags, Grid<Real>& div, MACGrid& vel){
		// ...
		if (flags.isFluid)
		{
			div(i,j,k) = vel(i+0.5,j,k)-vel(i-0.5,j,k) + vel(i,j+0.5,k) - vel(i,j-0.5,k);
		}
	}

	KERNEL(bnd = 1)
	void iterate_p(const Grid<Real>& div, Grid<Real>& pressure, const Grid<Real> &A0, const Grid<Real> &A1) {
		// ...
		//pressure equation is of the form Ap = d
		pressure(i,j,k) = 1*0.25/A1(i,j,k)*(div(i,j,k)-A1(i+1,j,k)*pressure(i+1,j,k)-A1(i,j+1,k)*pressure(i,j+1,k)-A0(i-1,j,k)*pressure(i-1,j,k)-A0(i,j-1,k)*pressure(i,j-1,k));

	}

	KERNEL(bnd = 1, reduce=+) returns(double residual=0.0)
	double iterate_r(const Grid<Real>& div, Grid<Real>& pressure, const Grid<Real> &A0, const Grid<Real> &A1) {
		// ...
		//residual is given by r = d- Ap(new)
		residual(i,j,k) = div(i,j,k) - -A1(i+1,j,k)*pressure(i+1,j,k)-A1(i,j+1,k)*pressure(i,j+1,k)-A0(i-1,j,k)*pressure(i-1,j,k)-A0(i,j-1,k)*pressure(i,j-1,k);

	}

	KERNEL(bnd = 1, reduce=+) returns(double residual=0.0)
	double iterate2(const Grid<Real>& div, Grid<Real>& pressure, const Grid<Real> &A0, const Grid<Real> &A1) {
		// ...
		residual(i,j,k) = div(i,j,k) - -A1(i+1,j,k)*pressure(i+1,j,k)-A1(i,j+1,k)*pressure(i,j+1,k)-A0(i-1,j,k)*pressure(i-1,j,k)-A0(i,j-1,k)*pressure(i,j-1,k);
		pressure(i,j,k) = residual(i,j,k)/A1(i,j,k);

		
	}

	KERNEL(bnd = 1)
	void UpdateVel(const FlagGrid& flags, MACGrid& vel, const Grid<Real>& pressure){
		// ...
		if (flags.isFluid)
		{
			vel(i,j,k) = vel(i,j,k) - Vec3(pressure(i-1,j,k)-pressure(i,j,k), pressure(i,j-1,k)-pressure(i,j,k)); 
		}
	}

	KERNEL(idx)
	void ComputeA0(const FlagGrid& flags, Grid<Real> &A0) {
		// ...
		if(flags.isFluid(i, j,k))
			A0(i,j,k) = -1;
		else 
			A0(i,j,k) = 0;
	}

	KERNEL(bnd = 1)
	void ComputeA1(const Grid<Real> &A0, Grid<Real> &A1) {
		// ...
		A1(i,j,k) = 0.0-(A0(i-1,j,k) + A0(i,j-1,k) + A0(i+1,j,k) + A0(i,j+1,k));
	}

	PYTHON() void solvePressureGS(FlagGrid& flags, MACGrid& vel, Grid<Real>& pressure, Real gsAccuracy = 1e-4) {
		// ...
		FluidSolver* parent = flags.getParent();
		Grid<Real> div(parent);
		Real res = 1.0;
		Grid<Real> residual(parent);
		Grid<Real> A0(parent);
		Grid<Real> A1(parent);

		ComputeDiv(flags, div, vel);
		ComputeA0(flags, A0);
		ComputeA1(A0, A1);

		while(res > gsAccuracy)
		{
			iterate_p(div, pressure, A0, A1);
			res = iterate_r(div, pressure, residual, A0, A1);
			//res = iterate2(div, pressure, residual, A0, A1);
		}
		UpdateVel(flags, vel, pressure);
	}

	PYTHON() Real getMaxDivergence(MACGrid& vel, FlagGrid& flags){
		// ...
		FluidSolver* parent = flags.getParent();
		Grid<Real> div(parent);
		return totalDivr(flags, div, vel)/totalNumber(div);
		return 0.;
	}

} // end namespace
