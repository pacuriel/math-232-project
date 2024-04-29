## notes from symeon
# stablility conditions 
# check timestep 
# spline with weird shape 
# start without obstructions (simple case) and check stability , accuracy (true solution and see what error is over time) ; compare with someone elses code 
# -> when bigger flow has more turbulance (symeons guess) 
## then add circles and understand what parameters you need for it to work 
### then can make a curly pipe with rectangular opening and ending 
# -> could be a a bunch of splines then a straight line 

# need to really understand what ngsolve does 

########################################################

# find analytical solution for rectangular domain with no obstructions and make a graph of error vs h 
# 1. analytical solution for pipe with no obstructions
# 2. plot errors w.r.t maxh
# 3. free errors compute approx order of convergence
# in 1D we take h1 = E1 & h2 = E2
# -> in 2D: N1 (number of nodes) = E1
# -> N2 (num nodes) = E2
# -> p is about log(E1/E2) / log(N2/N1)
# N1 is number of nodes with specified maxh and N2 is number of nodes with refined grid (smaller maxh)

# plot of N vs p
# what is CFL for NS?
# -> when refining h we need to refine delta t but making sure CFL remains constant

# do overall error and final time error (4 plots - error wrt h (2) and conv wrt h (2))
# -> use l max for overall and l2 for final time 

# one cylinder w vortex shedding
# if obstruction is [this big] what is the Reynolds number that will break it?
# domain cannot be symmetric otherwise it won't work 

# solving nonlinear term explicitly allows us to invert matrix to solve 
# then we solve linear terms implicitly