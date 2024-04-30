from ngsolve import *
from netgen.occ import *
import netgen.gui
from scipy.io import savemat
from scipy.interpolate import griddata
import numpy as np
import meshio
from configs import *
from tqdm import tqdm
import ngsolve.internal as ngsint
from netgen.geom2d import SplineGeometry
import matplotlib.pyplot as plt
import pickle
import os
import imageio


##################
### READ/WRITE ###
##################
def writeDict(d, fileName):
  with open(fileName, 'wb') as o: #convert to gzip
    pickle.dump(d, o, protocol=pickle.HIGHEST_PROTOCOL)

def readDict(fileName):
  d = {}
  try:
    with open(fileName, 'rb') as f:
      d = pickle.load(f)
  except FileNotFoundError: pass
  return d

#function to define the Geometry and return the mesh
def defineGeometry(domain_type='base', maxh=0.07):
    geo = SplineGeometry()
    if domain_type == 'base':
        geo.AddRectangle( (0, 0), (2, 0.41), bcs = ("wall", "outlet", "wall", "inlet"))
        geo.AddCircle ( (0.3, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl", maxh=0.02)
    elif domain_type == 'empty':
        geo.AddRectangle( (0, 0), (2, 0.41), bcs = ("wall", "outlet", "wall", "inlet"))
    elif domain_type == 'bulge':
        # river with a pond
        pnts =[(0,0), #1
               (0.5,0), #2
               (1,-0.52), #3
               (1.5,0), #4
               (2,0), #5
               (2, 0.41), #6
               (1.5,0.41), #7
               (1,0.91), #8
               (0.5,0.41), #9
               (0,0.41)] #10
        p1,p2,p3,p4,p5,p6,p7,p8,p9,p10 = [geo.AppendPoint(*pnt) for pnt in pnts]
        curves = [[["line",p1,p2],"wall"],
                  [["spline3",p2,p3,p4],"wall"],
                  [["line",p4,p5],"wall"],
                  [["line",p5,p6],"outlet"],  # out
                  [["line",p6,p7],"wall"],
                  [["spline3",p7,p8,p9],"wall"],
                  [["line",p9,p10],"wall"],
                  [["line",p10,p1],"inlet"]] # in
        geo.AddCircle((1, 0.2), r=0.2, leftdomain=0, rightdomain=1, bc="cyl", maxh=0.02)
        [geo.Append(c,bc=bc) for c,bc in curves]
    
    # rectangular obstructions (parallel)
    # geo.AddRectangle((0.1, 0.1), (0.2, 0.4), leftdomain=0, rightdomain=1, bc="rect_obstacle_1")
    # geo.AddRectangle((0.7, 0.1), (0.8, 0.3), leftdomain=0, rightdomain=1, bc="rect_obstacle_2")

    # rectangular obstructions (with a small opening)
    # geo.AddRectangle((0.7, 0.21), (0.75, 0.405), leftdomain=0, rightdomain=1, bc="rect_obstacle_1")
    # geo.AddRectangle((0.7, 0.01), (0.75, 0.17), leftdomain=0, rightdomain=1, bc="rect_obstacle_2")

    # staggered logs - need smaller timestep
    # geo.AddRectangle((0.2, 0.21), (0.25, 0.405), leftdomain=0, rightdomain=1, bc="rect_obstacle_1")
    # geo.AddRectangle((0.4, 0.01), (0.45, 0.17), leftdomain=0, rightdomain=1, bc="rect_obstacle_2")
    # geo.AddRectangle((0.6, 0.21), (0.65, 0.405), leftdomain=0, rightdomain=1, bc="rect_obstacle_3")
    # geo.AddRectangle((0.8, 0.01), (0.85, 0.17), leftdomain=0, rightdomain=1, bc="rect_obstacle_4")
    # geo.AddRectangle((1, 0.21), (1.05, 0.405), leftdomain=0, rightdomain=1, bc="rect_obstacle_5")
    # geo.AddRectangle((1.2, 0.01), (1.25, 0.17), leftdomain=0, rightdomain=1, bc="rect_obstacle_6")
    # geo.AddRectangle((1.4, 0.21), (1.45, 0.405), leftdomain=0, rightdomain=1, bc="rect_obstacle_7")
    # geo.AddRectangle((1.6, 0.01), (1.65, 0.17), leftdomain=0, rightdomain=1, bc="rect_obstacle_8")
    # geo.AddRectangle((1.8, 0.21), (1.85, 0.405), leftdomain=0, rightdomain=1, bc="rect_obstacle_9")

    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    
    return mesh

#function to simulate fluid flow with the given step size (maxh)
def simFlow(maxh, domain_type, nu, tau, tend):

    mesh = defineGeometry(domain_type=domain_type, maxh=maxh) #calling function to obtain mesh

    mesh.Curve(3)

    V = VectorH1(mesh,order=3, dirichlet="wall|cyl|inlet")  # third order basis functions 
    Q = H1(mesh,order=2)

    X = V*Q  # X is finite element function space 

    u,p = X.TrialFunction()
    v,q = X.TestFunction()

    stokes = nu*InnerProduct(grad(u), grad(v))+div(u)*q+div(v)*p - 1e-10*p*q
    a = BilinearForm(X, symmetric=True)
    a += stokes*dx
    a.Assemble()

    # nothing here ...
    f = LinearForm(X)   
    f.Assemble()

    # gridfunction for the solution
    gfu = GridFunction(X)

    # setting parabolic inflow at inlet: 
    uin = CoefficientFunction( (1.5*4*y*(0.41-y)/(0.41*0.41), 0) )  # "exact" analytical solution to poiseuille equation, format=(u_x, u_y)
    gfu.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))  # boundary conditions - 0 for velocity and 1 for pressure

    # solve Stokes problem for initial conditions:
    inv_stokes = a.mat.Inverse(X.FreeDofs())

    res = f.vec.CreateVector()
    res.data = f.vec - a.mat*gfu.vec
    gfu.vec.data += inv_stokes * res

    # matrix for implicit Euler 
    mstar = BilinearForm(X, symmetric=True)
    mstar += (u*v + tau*stokes)*dx
    mstar.Assemble()
    inv = mstar.mat.Inverse(X.FreeDofs(), inverse="sparsecholesky")

    # the non-linear term 
    conv = BilinearForm(X, nonassemble = True)
    conv += (grad(u) * u) * v * dx

    # for visualization
    Draw (Norm(gfu.components[0]), mesh, "velocity", sd=3)  # 0 for velocity, 1 for pressure
    ngsint.viewoptions.drawoutline=1 # disable triangle outline when plotting
    # ngsint.VideoStart('snapshits\\test.mp4')
    # implicit Euler/explicit Euler splitting method:
    t = 0
    with TaskManager():
        while t < tend:
            print ("t=", t, end="\r")

            conv.Apply (gfu.vec, res)
            res.data += a.mat*gfu.vec
            gfu.vec.data -= tau * inv * res    

            t = t + tau
            t_str = str(t)
            # ngsint.VideoAddFrame()
            Redraw()
            ngsint.SnapShot('snap_shits/test_' + t_str)

            #checking if last time step and obtaining solution
            if t >= tend:
                computed_soln = np.zeros(mesh.nv)
                true_soln = np.zeros(mesh.nv)
                i = 0 #iterator
                for p in mesh.ngmesh.Points(): 
                    x_pt = p[0] 
                    y_pt = p[1]
                    curr_pt = mesh(x_pt, y_pt) #current point
                    computed_soln[i] = gfu.components[0][0](curr_pt) #obtaining computed soln
                    true_soln[i] = (1.5*4*y_pt*(0.41-y_pt))/(0.41*0.41)
                    i = i + 1 #incrementing iterator

        # ngsint.VideoFinalize()

    return gfu, mesh, computed_soln, true_soln

#plotting errors
def plotErrors(errors, maxh_vals, node_cts):
    N = len(maxh_vals)
    error_ratio = [np.NaN]
    orders = [np.NaN]
    for i in range(1,N):
        error_ratio.append(errors[i-1] / errors[i])
        orders.append(np.log(abs(error_ratio[i])) / np.log(abs(node_cts[i-1] / node_cts[i])))

    #displaying order plot
    print(f"Node counts\t\tError\t\t\tRatio\t\t\tObserved order")
    for i in range(N):
        print(f"{node_cts[i]}\t\t{errors[i]}\t\t{error_ratio[i]}\t\t{orders[i]}")    

    #plotting
    # sns.regplot(x=node_cts, y=errors)
    # plt.show()
    # plt.plot(node_cts, errors)
    # plt.xlabel('Node counts $N$')
    # plt.ylabel('Error')
    # plt.title('Number of nodes vs. Error')
    # plt.show()

#driver function
def main():

    nu = 0.001 #viscosity
    # timestepping parameters
    tau = 0.001 #timestep
    tend = 5 #end time
    domain_type = 'base' # mesh

    ###trying to make convergence plot
    # base_maxh = 0.0175
    # maxh_vals = [(2**i)*base_maxh for i in range(0,6)]
    maxh_vals = [0.07]
    errors = [] #used to store
    node_cts = []

    if not os.path.exists('data\\data_dict.pickle') or os.path.exists('data\\data_dict.pickle'):
        for maxh in maxh_vals:
            gfu, mesh, computed_soln, true_soln = simFlow(maxh, domain_type, nu, tau, tend)

            # print(f"maxh = {maxh}")
            error = np.linalg.norm(computed_soln - true_soln, ord=np.inf)
            errors.append(error)
            node_cts.append(mesh.nv)

            # print(f"For maxh = {maxh}, error = {np.linalg.norm(computed_soln - true_soln, ord=np.inf)}")
        data_dict = {'errors': errors, 'maxh_vals': maxh_vals, 'node_cts': node_cts}
        writeDict(data_dict, 'data\\data_dict.pickle')
    else:
        data_dict = readDict('data\\data_dict.pickle')
        errors = data_dict['errors']
        node_cts = data_dict['node_cts']
        plotErrors(errors, maxh_vals, node_cts) #plotting errors


    images = []
    for filename in os.listdir('snap_shits/'):
        images.append(imageio.imread('snap_shits\\' + filename))

    imageio.mimsave('test.gif', images)

if __name__ == "__main__":
    main()