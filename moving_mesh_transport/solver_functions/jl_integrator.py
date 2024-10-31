# using IJulia
# IJulia.installkernel("Julia nodeps", "--depwarn=no")
import juliacall
# IJulia =  juliacall.newmodule("IJulia")
# # jl = juliacall.newmodule("Pkg")
# IJulia.installkernel("Julia nodeps", "--depwarn=no")
# jl.installkernel("Julia nodeps", "--depwarn=no")

from diffeqpy import ode
# import diffeqpy
# from diffeqpy import ode



def integrator(RHS, IC, tspan, t_eval):

    def f(u,p,t):
        return RHS


    prob = ode.ODEProblem(f, IC, tspan)
    assert(0)
    sol = ode.solve(prob, ode.Rosenbrock23(), saveat = t_eval)
    return sol