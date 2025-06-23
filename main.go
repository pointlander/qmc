// Copyright 2025 The QMC Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
)

var (
	// FlagOriginal the original model
	FlagOriginal = flag.Bool("original", false, "the original model")
)

// Original the original model
func Original() {
	rng := rand.New(rand.NewSource(1))
	// potential energy function
	V := func(x float64) float64 {
		// use units such that m = 1 and omega_0 = 1
		return 0.5 * math.Pow(x, 2.0)
	}

	// derivative dV(x)/dx used in virial theorem
	dVdx := func(x float64) float64 {
		return x
	}

	var (
		tau       float64   // imaginary time period
		M         int       // number of time slices
		Delta_tau float64   // imaginary time step
		x         []float64 // displacements from equilibrium of M "atoms"
		n_bins    int       // number of bins for psi histogram
		x_min     float64   // bottom of first bin
		x_max     float64   // top of last bin
		dx        float64   // bin width
		P         []float64 // histogram for |psi|^2
		delta     float64   // Metropolis step size in x
		MC_steps  int       // number of Monte Carlo steps in simulation
	)

	initialize := func() {
		Delta_tau = tau / float64(M)
		x = make([]float64, M)
		x_min = -x_max
		dx = (x_max - x_min) / float64(n_bins)
		P = make([]float64, n_bins)
		fmt.Println(" Initializing atom positions using gsl::ran_uniform()")
		for j := 0; j < M; j++ {
			x[j] = (2*rng.Float64() - 1) * x_max
		}
	}

	Metropolis_step_accepted := func(x_new *float64) bool {
		// choose a time slice at random
		j := int(rng.Float64() * float64(M))
		// indexes of neighbors periodic in tau
		j_minus, j_plus := j-1, j+1
		if j_minus < 0 {
			j_minus = M - 1
		}
		if j_plus > M-1 {
			j_plus = 0
		}
		// choose a random trial displacement
		x_trial := x[j] + (2*rng.Float64()-1)*delta
		// compute change in energy
		Delta_E := V(x_trial) - V(x[j]) +
			0.5*math.Pow((x[j_plus]-x_trial)/Delta_tau, 2.0) +
			0.5*math.Pow((x_trial-x[j_minus])/Delta_tau, 2.0) -
			0.5*math.Pow((x[j_plus]-x[j])/Delta_tau, 2.0) -
			0.5*math.Pow((x[j]-x[j_minus])/Delta_tau, 2.0)
		if Delta_E < 0.0 || math.Exp(-Delta_tau*Delta_E) > rng.Float64() {
			x[j] = x_trial
			*x_new = x_trial
			return true
		} else {
			*x_new = x[j]
			return false
		}
	}

	fmt.Println(" Path Integral Monte Carlo for the Harmonic Oscillator")
	fmt.Println(" -----------------------------------------------------")
	// set simulation parameters
	tau = 10.0
	fmt.Println(" Imaginary time period tau = ", tau)
	M = 100
	fmt.Println(" Number of time slices M = ", M)
	x_max = 4.0
	fmt.Println(" Maximum displacement to bin x_max = ", x_max)
	n_bins = 100
	fmt.Println(" Number of histogram bins in x = ", n_bins)
	delta = 1.0
	fmt.Println(" Metropolis step size delta = ", delta)
	MC_steps = 100000
	fmt.Println(" Number of Monte Carlo steps = ", MC_steps)
	initialize()
	therm_steps := MC_steps / 5
	acceptances := 0
	x_new := 0.0
	fmt.Printf(" Doing %v thermalization steps ...", therm_steps)

	for step := 0; step < therm_steps; step++ {
		for j := 0; j < M; j++ {
			if Metropolis_step_accepted(&x_new) {
				acceptances++
			}
		}
	}

	fmt.Println(" Percentage of accepted steps = ", float64(acceptances)/float64(M*therm_steps)*100.0)
	E_sum := 0.0
	E_sqd_sum := 0.0
	for i := range P {
		P[i] = 0
	}
	acceptances = 0
	fmt.Println(" Doing ", MC_steps, " production steps ...")
	for step := 0; step < MC_steps; step++ {
		for j := 0; j < M; j++ {
			if Metropolis_step_accepted(&x_new) {
				acceptances++
			}
			// add x_new to histogram bin
			bin := int((x_new - x_min) / (x_max - x_min) * float64(n_bins))
			if bin >= 0 && bin < M {
				P[bin] += 1
			}
			// compute Energy using virial theorem formula and accumulate
			E := V(x_new) + 0.5*x_new*dVdx(x_new)
			E_sum += E
			E_sqd_sum += E * E
		}
	}

	// compute averages
	values := MC_steps * M
	E_ave := E_sum / float64(values)
	E_var := E_sqd_sum/float64(values) - E_ave*E_ave
	fmt.Println("<E> = ", E_ave, " +/- ", math.Sqrt(E_var/float64(values)))
	fmt.Println(" <E^2> - <E>^2 = ", E_var)
	of, err := os.Create("pimc.out")
	if err != nil {
		panic(err)
	}
	defer of.Close()
	E_ave = 0
	for bin := 0; bin < n_bins; bin++ {
		x := x_min + dx*(float64(bin)+0.5)
		fmt.Fprintf(of, " %v\t%v\n", x, P[bin]/float64(values))
		E_ave += P[bin] / float64(values) * (0.5*x*dVdx(x) + V(x))
	}
	fmt.Println(" <E> from P(x) = ", E_ave)
	fmt.Println(" Probability histogram written to file pimc.out")
}

func main() {
	flag.Parse()

	if *FlagOriginal {
		Original()
		return
	}
}
