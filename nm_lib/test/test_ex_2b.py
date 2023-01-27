def test_ex_2b():
    """
    Test to see if the numerical `evolv_adv_burgers` method
    gives a good enough approximation with a tolerance of 
    1e-5
    """
    from nm_lib import nm_lib as nm
    import numpy as np 
    
    def u_init(x): 
        return np.cos(6*np.pi*x/5)**2 / np.cosh(5*x**2)
    
    def u(x, t, a=-1): 
        return u_init(x - a*t)

    x0 = -2.6
    xf = -1*x0
    nump = 65
    xx = np.arange(nump)/(nump - 1) * (xf - x0) + x0

    nt = 100 

    U0 = u_init(xx)
    t, calculated = nm.evolv_adv_burgers(xx, U0, nt, a=1)
    
    expected = u(np.repeat(xx[:, np.newaxis], nt, axis=1),
                np.repeat(t[np.newaxis, :], nump, axis=0), a=1)

    tol = np.ones(len(xx))*1e-5

    success = np.any(np.abs(calculated[:, 0] - expected[:, 0]) < tol)

    assert success
