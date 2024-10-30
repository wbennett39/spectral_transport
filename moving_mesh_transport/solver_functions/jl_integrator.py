from diffeqpy import de



def integrator(RHS, IC, tspan, t_eval):
    prob = de.ODEProblem(RHS, IC, tspan)
    sol = de.solve(prob, de.Rosenbrock23())
    return sol