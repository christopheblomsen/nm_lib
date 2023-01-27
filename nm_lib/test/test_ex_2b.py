def test_ex_2b():
    """
    Test to see if the numerical `evolv_adv_burgers` method gives a good enough
    approximation with a tolerance of 1e-5.
    """
    import numpy as np

    from nm_lib import nm_lib as nm

    def u_init(x):
        """
        The initial function at t=t0.

        Parameters
        ----------
        x   :   `array`
             Spatial axis

        Returns
        -------
        `array`
            The initial function at t=t0
        """
        return np.cos(6 * np.pi * x / 5) ** 2 / np.cosh(5 * x**2)

    def u(x, t, a=-1):
        """
        Analytical solution to the equation.

        Parameters
        -----------
        x   :   `array`
             Spatial axis
        t   :   `array`
             Time axis
        a : `float` or `array`
            Either constant, or array which multiply the right hand side of the Burger's eq.

        Returns
        -------
        `array`
            Analytical solution to burgers equations
        """
        return u_init(x - a * t)

    x0 = -2.6
    xf = -1 * x0
    nump = 65
    xx = np.arange(nump) / (nump - 1) * (xf - x0) + x0

    nt = 100

    U0 = u_init(xx)
    t, calculated = nm.evolv_adv_burgers(xx, U0, nt, a=1)

    expected = u(
        np.repeat(xx[:, np.newaxis], nt, axis=1),
        np.repeat(t[np.newaxis, :], nump, axis=0),
        a=1,
    )

    tol = np.ones(len(t)) * 1e-10
    diff = np.abs(calculated - expected)
    success = np.any(diff < tol)
    msg = np.max(diff)
    assert success, msg
