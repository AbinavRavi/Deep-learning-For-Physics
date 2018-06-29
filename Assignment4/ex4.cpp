/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2011 Tobias Pfaff, Nils Thuerey 
 *
 * This program is free software, distributed under the terms of the
 * GNU General Public License (GPL) 
 * http://www.gnu.org/licenses
 *
 ******************************************************************************/
#include <thread>
#include "kernel.h"
#include "grid.h"
#include "pcgsolver.h"

using namespace std;
namespace Manta {
//4.2
KERNEL(bnd = 1) void CalculatediffuseTemperatureExplicit(FlagGrid& flags, Grid<Real>&T0,Real alpha)
{
	Real dt = flags.getParent()->getDt();
	if(flags.isFluid(i,j,k)){
		T0(i,j,k) = T0(i,j,k) + dt * alpha * (T0(i+1,j+1,k) + T0(i-1,j-1,k) - 4 * T0(i,j,k) + T0(i-1,j+1,k) + T0(i+1,j-1,k));
	}
	else{
		T0(i,j,k) = 0;
	} 

}
PYTHON() void diffuseTemperatureExplicit(FlagGrid& flags, Grid<Real>& temperature, Real alpha){
	// don't overwrite values in T that will be read again
	// write a KERNEL and make sure that the temperature in boundary cells stays zero
	CalculatediffuseTemperatureExplicit(flags,temperature,alpha);
}

// 4.3
KERNEL() void setupB(Grid<Real>& T0,std::vector<Real>* b){ 
	(*b)[i*T0.getStrideX()+j*T0.getStrideY()] = T0(i,j,k);
}

KERNEL() void fillT(FlagGrid& flags,std::vector<Real>& x,Grid<Real>& T){
	// make sure that temperature values in boundary cells are zero!
	if(flags.isFluid(i,j,k))
		T(i,j,k) = x[ i*flags.getStrideX()+j*flags.getStrideY()];
	else
		T(i,j,k) = 0;
}

// use the single argument to prevent multithreading (multiple threads might interfere during the matrix setup)
KERNEL(single) void setupA(FlagGrid& flags, SparseMatrix<Real>& A, int N, Real alpha, Real dt){  
	// set with:  A.set_element( index1, index2 , value );
	// if needed, read with: A(index1, index2);

		// avoid zero rows in A -> set the diagonal value for boundary cells to 1.0
	if(flags.isFluid(i,j,k))
	{
		A.set_element( i*flags.getStrideX()+j*flags.getStrideY(), i*flags.getStrideX()+j*flags.getStrideY() , 1.0+4*alpha*dt );

		A.set_element( i*flags.getStrideX()+j*flags.getStrideY(), i+1*flags.getStrideX()+j*flags.getStrideY() , -alpha*dt );
		A.set_element( i*flags.getStrideX()+j*flags.getStrideY(), i-1*flags.getStrideX()+j*flags.getStrideY() , -alpha*dt );
		A.set_element( i*flags.getStrideX()+j*flags.getStrideY(), i*flags.getStrideX()+j+1*flags.getStrideY() , -alpha*dt );
		A.set_element( i*flags.getStrideX()+j*flags.getStrideY(), i*flags.getStrideX()+j-1*flags.getStrideY() , -alpha*dt );
	}
	else
		A.set_element( i*flags.getStrideX()+j*flags.getStrideY(), i*flags.getStrideX()+j*flags.getStrideY() , 1.0 );
}

PYTHON() void diffuseTemperatureImplicit(FlagGrid& flags, Grid<Real>& T0, Real alpha, Real dt){
	// solve A T = b
	const int N = T0.getSizeX()*T0.getSizeY();
	SparseMatrix<Real> A(N);
	std::vector<Real> b(N);

	setupA(flags, A, N, alpha,dt);
	setupB(T0, &b);

	// perform solve
	Real pcg_target_residual = 1e-05;
	Real pcg_max_iterations = 1000;
	Real ret_pcg_residual = 1e10;
	int  ret_pcg_iterations = -1;

	SparsePCGSolver<Real> solver;
	solver.set_solver_parameters(pcg_target_residual, pcg_max_iterations, 0.97, 0.25);

	std::vector<Real> x(N);
	for (int j = 0; j<N; ++j) { x[j] = 0.; }

	// preconditioners: 0 off, 1 diagonal, 2 incomplete cholesky
	solver.solve(A, b, x, ret_pcg_residual, ret_pcg_iterations, 0);

	// x contains the new temperature values
	fillT(flags, x, T0);
}

}; 