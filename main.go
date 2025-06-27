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
	// FlagIsing the ising model
	FlagIsing = flag.Bool("ising", false, "the ising model")
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

// Ising is the ising model
func Ising() {
	rng := rand.New(rand.NewSource(1))

	//----------------------------------------------------------------------
	//  BLOCK OF FUNCTIONS USED IN THE MAIN CODE
	//----------------------------------------------------------------------
	initialstate := func(N int) [][]float64 {
		// generates a random spin configuration for initial condition
		state := make([][]float64, N)
		for i := range state {
			state[i] = make([]float64, N)
			for ii := range state[i] {
				state[i][ii] = float64(2*rng.Intn(2) - 1)
			}
		}
		return state
	}

	mcmove := func(config [][]float64, beta float64) [][]float64 {
		// Monte Carlo move using Metropolis algorithm
		N := len(config)
		for i := range config {
			for range config[i] {
				a := rng.Intn(N)
				b := rng.Intn(N)
				s := config[a][b]
				nb := config[(a+1)%N][b] + config[a][(b+1)%N] + config[(a-1+N)%N][b] + config[a][(b-1+N)%N]
				cost := 2 * s * nb
				if cost < 0 {
					s *= -1
				} else if rng.Float64() < math.Exp(-cost*beta) {
					s *= -1
				}
				config[a][b] = s
			}
		}
		return config
	}

	calcEnergy := func(config [][]float64) float64 {
		// Energy of a given configuration
		N := len(config)
		energy := 0.0
		for i := range config {
			for j := range config[i] {
				S := config[i][j]
				nb := config[(i+1)%N][j] + config[i][(j+1)%N] + config[(i-1+N)%N][j] + config[i][(j-1+N)%N]
				energy += -nb * S
			}
		}
		return energy / 4.
	}

	calcMag := func(config [][]float64) float64 {
		// Magnetization of a given configuration
		mag := 0.0
		for i := range config {
			for _, value := range config[i] {
				mag += value
			}
		}
		return mag
	}

	// change these parameters for a smaller (faster) simulation
	nt := 88        //  number of temperature points
	N := 16         //  size of the lattice, N x N
	eqSteps := 1024 //  number of MC sweeps for equilibration
	mcSteps := 1024 //  number of MC sweeps for calculation

	T := make([]float64, nt)
	start := 1.53
	stop := 3.28
	dx := (stop - start) / float64(nt)
	for i := range T {
		T[i] = start + float64(i)*dx
	}
	E, M, C, X := make([]float64, nt), make([]float64, nt), make([]float64, nt), make([]float64, nt)
	n1, n2 := 1.0/float64(mcSteps*N*N), 1.0/float64(mcSteps*mcSteps*N*N)
	// divide by number of samples, and by system size to get intensive values

	//----------------------------------------------------------------------
	//  MAIN PART OF THE CODE
	//----------------------------------------------------------------------
	for tt := range nt {
		E1 := 0.0
		M1 := 0.0
		E2 := 0.0
		M2 := 0.0
		config := initialstate(N)
		iT := 1.0 / T[tt]
		iT2 := iT * iT

		for range eqSteps { // equilibrate
			mcmove(config, iT) // Monte Carlo moves
		}

		for range mcSteps {
			mcmove(config, iT)
			Ene := calcEnergy(config) // calculate the energy
			Mag := calcMag(config)    // calculate the magnetisation

			E1 = E1 + Ene
			M1 = M1 + Mag
			M2 = M2 + Mag*Mag
			E2 = E2 + Ene*Ene
		}

		E[tt] = n1 * E1
		M[tt] = n1 * M1
		C[tt] = (n1*E2 - n2*E1*E1) * iT2
		X[tt] = (n1*M2 - n2*M1*M1) * iT
	}

	fmt.Println(E)
	fmt.Println(M)
}

// S translates -1 to 0 and 1 to 1
func S(a float64) uint {
	if a == -1.0 {
		return 0
	}
	return 1
}

// Electron is an electron with spin
type Electron struct {
	Spin  float64
	Links []*Electron
}

// System is a system of electrons
type System struct {
	Rng       *rand.Rand
	Electrons []*Electron
}

// NewSystem creates a new system of electrons
func NewSystem(n int64) System {
	return System{
		Rng:       rand.New(rand.NewSource(n)),
		Electrons: make([]*Electron, n),
	}
}

// Link adds a link between to electrons
func (s *System) Link(a, b int) {
	s.Electrons[a].Links = append(s.Electrons[a].Links, s.Electrons[b])
}

// Step steps the mode
func (s *System) Step(beta float64, c Cost) {
	// Monte Carlo move using Metropolis algorithm
	N := len(s.Electrons)
	for range s.Electrons {
		a := s.Rng.Intn(N)
		e := s.Electrons[a]
		histogram := [2]float64{}
		for _, e := range e.Links {
			histogram[S(e.Spin)]++
		}
		cost := 2 * e.Spin * c(histogram)
		if cost < 0 {
			e.Spin *= -1
		} else if s.Rng.Float64() < math.Exp(-cost*beta) {
			e.Spin *= -1
		}
	}
}

// CalcEnergy calculates the energy
func (s *System) CalcEnergy() float64 {
	// Energy of a given configuration
	energy := 0.0
	for _, e := range s.Electrons {
		S := e.Spin
		nb := 0.0
		for _, e := range e.Links {
			nb += e.Spin
		}
		energy += -nb * S
	}
	return energy / 4.0
}

// CalcMag calculates the magnetization
func (s *System) CalcMag() float64 {
	// Magnetization of a given configuration
	mag := 0.0
	for i := range s.Electrons {
		mag += s.Electrons[i].Spin
	}
	return mag
}

// Cost the cost
type Cost func(histogram [2]float64) float64

// Entropy use entropy for cost
func Entropy(histogram [2]float64) float64 {
	sum := 0.0
	for _, value := range histogram {
		sum += value
	}
	entropy := 0.0
	for _, value := range histogram {
		if value == 0 {
			continue
		}
		entropy += (value / sum) * math.Log2(value/sum)
	}
	return -entropy
}

// Spin use spin for cost
func Spin(histogram [2]float64) float64 {
	return histogram[1] - histogram[0]
}

// Model is the model
func Model(c Cost) {
	rng := rand.New(rand.NewSource(1))

	// change these parameters for a smaller (faster) simulation
	nt := 88        //  number of temperature points
	N := 16         //  size of the lattice, N x N
	eqSteps := 1024 //  number of MC sweeps for equilibration
	mcSteps := 1024 //  number of MC sweeps for calculation

	T := make([]float64, nt)
	start := 1.53
	stop := 3.28
	dx := (stop - start) / float64(nt)
	for i := range T {
		T[i] = start + float64(i)*dx
	}
	E, M, C, X := make([]float64, nt), make([]float64, nt), make([]float64, nt), make([]float64, nt)
	n1, n2 := 1.0/float64(mcSteps*N*N), 1.0/float64(mcSteps*mcSteps*N*N)
	// divide by number of samples, and by system size to get intensive values

	histogram := [2]float64{}
	for tt := range nt {
		E1 := 0.0
		M1 := 0.0
		E2 := 0.0
		M2 := 0.0
		config := NewSystem(5)
		for i := range config.Electrons {
			e := Electron{
				Spin: float64(2*rng.Intn(2) - 1),
			}
			config.Electrons[i] = &e
		}
		config.Link(0, 1)
		config.Link(0, 3)
		config.Link(1, 0)
		config.Link(1, 2)
		config.Link(2, 1)
		config.Link(2, 3)
		config.Link(2, 4)
		config.Link(3, 0)
		config.Link(3, 2)
		config.Link(3, 4)
		config.Link(4, 2)
		config.Link(4, 3)

		iT := 1.0 / T[tt]
		iT2 := iT * iT

		for range eqSteps { // equilibrate
			config.Step(iT, c) // Monte Carlo moves
		}

		for range mcSteps {
			config.Step(iT, c)
			Ene := config.CalcEnergy() // calculate the energy
			Mag := config.CalcMag()    // calculate the magnetisation
			histogram[S(config.Electrons[4].Spin)]++
			E1 = E1 + Ene
			M1 = M1 + Mag
			M2 = M2 + Mag*Mag
			E2 = E2 + Ene*Ene
		}

		E[tt] = n1 * E1
		M[tt] = n1 * M1
		C[tt] = (n1*E2 - n2*E1*E1) * iT2
		X[tt] = (n1*M2 - n2*M1*M1) * iT
	}

	fmt.Println(E)
	fmt.Println(M)
	fmt.Println(histogram)
}

func main() {
	flag.Parse()

	if *FlagOriginal {
		Original()
		return
	}

	if *FlagIsing {
		Ising()
		return
	}

	fmt.Println("Spin")
	Model(Spin)
	fmt.Println("Entropy")
	Model(Entropy)
}
