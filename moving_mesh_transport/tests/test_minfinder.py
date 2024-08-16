import numpy as np

# the minimum value

def randomfunc(x):
     return (-30 + x)**2 - (-np.exp(-x/2.) + np.sin(x)**2)/np.sqrt(-10 + x)
     

def find_minimum(a=20.5, b=55.5):
        dx = (b-a)/2
        pool_size = 5
        npts = 31
        npts2 = 31
        converged = False
        tol = 1e-15

        initial_guess = np.linspace(a,b,npts)
        ee = randomfunc(initial_guess)
        emins_initial = np.sort(ee)[0:pool_size]
        xvals = np.zeros(pool_size)
        emins = np.zeros(pool_size)
        emins = emins_initial
        for n in range(pool_size):
            xvals[n] = initial_guess[np.argmin(np.abs(ee-emins_initial[n]))]


        it = 0
        # while converged == False:

        emins_old = emins
        for ix in range(pool_size):
            # xs = np.linspace(xvals[ix]-dx, xvals[ix]+dx, npts2)

            # ee = randomfunc(xs)
            # emin = np.sort(ee)[0]

            # xval = xs[np.argmin(np.abs(ee-emin))]
            # # if emins[ix] == emin:
            # #      assert(0)
            # # else:
            # print(emins[ix], emin)
            xvals[ix] = gradient_descent(xvals[ix]-dx, xvals[ix]+dx, xvals[ix], randomfunc)
            
            emins[ix] = randomfunc(xvals[ix])
            # xvals[ix] = xval

        dx = dx/ 2
        it += 1
        
        # if np.max(np.abs(emins_old - emins)) <= tol:
        # if np.abs(emins_old[0] - emins[0]) < tol:
        #         converged = True
        #         print('converged')
        #         print(emins_old,'old')
        #         print(emins, 'emin')
        #         print(np.max(np.abs(emins_old - emins)))
            

        # if (np.sort(emins)[0] > emins_initial).any():
        #      print(emins, 'min vals')
        #      print(emins_initial, 'initial min values')
            #  assert 0
        # print(emins)
        return np.sort(emins)[0]

def test_find_minimum():
     res = find_minimum()
     assert np.abs(res- -0.21940365973414816) <=1e-6


def gradient_descent(a,b,x0, f):
     step = (b-a)/2
     tol = step/5000
     loc = x0
     loc_old = loc
     direction = 1.0
     while step >tol:
          loc += step * direction
          if f(loc) > f(loc_old):
               step = step/2.0

               direction = direction * -1
          loc_old = loc

     return loc

          
