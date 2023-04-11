#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 02 10:25:17 2021.

@author: Juan Martinez Sykora
"""

# import builtin modules

# import external public "common" modules
import numpy as np


def deriv_dnw(xx, hh, **kwargs):
    """
    Returns the downwind 2nd order derivative of hh array respect to xx array.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.

    Returns
    -------
    `array`
        The downwind 2nd order derivative of hh respect to xx. Last
        grid point is ill (or missing) calculated.
    """
    # Using the roll method
    y = (np.roll(hh, -1) - np.roll(hh, 0)) / (np.roll(xx, -1) - np.roll(xx, 0))
    """
    # Using the classical method
    N = len(xx)
    xh = np.zeros(N)

    y = np.zeros(N)

    xh[0] = xx[0] - 0.5 * (xx[1] - xx[0])
    y[0] = hh[0]

    for i in range(N - 1):
        xh[i + 1] = 0.5 * (xx[i + 1] + xx[i])

        y[i + 1] = (hh[i + 1] - hh[i]) / (xx[i + 1] - xx[i])

    xh[-1] = xx[-1] + 0.5 * (xx[-1] - xx[-2])
    y[-1] = hh[-1]
    """
    return y


def order_conv(hh, hh2, hh4, **kwargs):
    """
    Computes the order of convergence of a derivative function.

    Parameters
    ----------
    hh : `array`
        Function that depends on xx.
    hh2 : `array`
        Function that depends on xx but with twice number of grid points than hh.
    hh4 : `array`
        Function that depends on xx but with twice number of grid points than hh2.
    Returns
    -------
    `array`
        The order of convergence.
    """
    return np.ma.log2((hh4[::4] - hh2[::2])/(hh2[::2] - hh))


def deriv_4tho(xx, hh, **kwargs):
    """
    Returns the 4th order derivative of hh respect to xx.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.

    Returns
    -------
    `array`
        The centered 4th order derivative of hh respect to xx.
        Last and first two grid points are ill calculated.
    """
    N = len(xx)
    y = np.zeros(N)

    # dx = np.gradient(xx)
    dx = xx[1] - xx[0]

    y[0] = hh[0]
    y[1] = hh[1]

    for i in range(2, N - 2):
        y[i] = (hh[i - 2] - 8 * hh[i - 1] + 8 * hh[i + 1] - hh[i + 2]) / (12 * dx)

    return y


def step_adv_burgers(
    xx, hh, a, cfl_cut=0.98, ddx=lambda x, y: deriv_dnw(x, y), **kwargs
):
    r"""
    Right hand side of Burger's eq. where a can be a constant or a function
    that depends on xx.

    Requires
    ----------
    cfl_adv_burger function which computes np.min(dx/a)

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
        By default clf_cut=0.98.
    ddx : `lambda function`
        Allows to select the type of spatial derivative.
        By default lambda x,y: deriv_dnw(x, y)

    Returns
    -------
    `array`
        Time interval.
        Right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x}
    """
    dt = cfl_cut * cfl_adv_burger(a, xx)
    u_new = a*ddx(xx, hh)

    return dt, u_new


def cfl_adv_burger(a, x):
    """
    Computes the dt_fact, i.e., Courant, Fredrich, and Lewy condition for the
    advective term in the Burger's eq.

    Parameters
    ----------
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    x : `array`
        Spatial axis.

    Returns
    -------
    `float`
        min(dx/|a|)
    """
    dx = np.gradient(x)
    return np.min(dx / np.abs(a))


def evolv_adv_burgers(
    xx,
    hh,
    nt,
    a,
    cfl_cut=0.98,
    ddx=lambda x, y: deriv_dnw(x, y),
    bnd_type="wrap",
    bnd_limits=[0, 1],
    **kwargs,
):
    r"""
    Advance nt time-steps in time the burger eq for a being a a fix constant or array.
    Requires
    ----------
    step_adv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_dnw(x, y).
    bnd_type : `string`
        Allows to select the type of boundaries.
        By default 'wrap'.
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [0,1].

    Returns
    -------
    t : `array`
        time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    """
    N = np.size(xx)

    unnt = np.zeros((N, nt))
    unnt[:, 0] = hh

    t = np.zeros(nt)

    for i in range(0, nt - 1):
        dt, tmp = step_adv_burgers(xx, unnt[:, i], a, cfl_cut=cfl_cut, ddx=ddx)
        t[i + 1] = t[i] + dt
        tmmp = unnt[:, i] - tmp * dt
        # For upwind and centre
        if bnd_limits[1] > 0:
            unnt[:, i + 1] = np.pad(
                tmmp[bnd_limits[0]: -bnd_limits[1]], bnd_limits, bnd_type
            )
        # For downwind
        else:
            unnt[:, i + 1] = np.pad(tmmp[bnd_limits[0]:], bnd_limits, bnd_type)

    return t, unnt


def deriv_upw(xx, hh, **kwargs):
    r"""
    returns the upwind 2nd order derivative of hh respect to xx.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.

    Returns
    -------
    `array`
        The upwind 2nd order derivative of hh respect to xx. First
        grid point is ill calculated.
    """
    # Using the classical method
    """
    N = len(xx)
    xh = np.zeros(N)

    y = np.zeros(N)

    xh[0] = xx[0] - 0.5 * (xx[1] - xx[0])
    y[0] = hh[0]

    for i in range(N):
        #xh[i + 1] = 0.5 * (xx[i + 1] + xx[i])

        y[i] = (hh[i] - hh[i-1]) / (xx[i] - xx[i - 1])

    xh[-1] = xx[-1] + 0.5 * (xx[-1] - xx[-2])
    y[-1] = hh[-1]
    """
    # Using the roll method
    y = (np.roll(hh, 0) - np.roll(hh, 1)) / (np.roll(xx, 0) - np.roll(xx, 1))
    return y


def deriv_cent(xx, hh, **kwargs):
    r"""
    returns the centered 2nd derivative of hh respect to xx.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.

    Returns
    -------
    `array`
        The centered 2nd order derivative of hh respect to xx. First
        and last grid points are ill calculated.
    """
    return (np.roll(hh, -1) - np.roll(hh, 1)) / (2 * (np.roll(xx, -1) - np.roll(xx, 1)))


def evolv_uadv_burgers(
    xx,
    hh,
    nt,
    cfl_cut=0.98,
    ddx=lambda x, y: deriv_dnw(x, y),
    bnd_type="wrap",
    bnd_limits=[0, 1],
    **kwargs,
):
    r"""
    Advance nt time-steps in time the burger eq for a being u.

    Requires
    --------
    step_uadv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    cfl_cut : `float`
        constant value to limit dt from cfl_adv_burger.
        By default 0.98.
    ddx : `lambda function`
        Allows to change the space derivative function.
    bnd_type : `string`
        It allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    """
    N = np.size(xx)

    unnt = np.zeros((N, nt))
    unnt[:, 0] = hh

    t = np.zeros(nt)

    for i in range(0, nt - 1):
        dt, tmp = step_adv_burgers(xx, unnt[:, i], unnt[:, i], cfl_cut=cfl_cut, ddx=ddx)
        t[i + 1] = t[i] + dt
        tmmp = unnt[:, i] - tmp * dt
        # For upwind and centre
        if bnd_limits[1] > 0:
            unnt[:, i + 1] = np.pad(
                tmmp[bnd_limits[0]: -bnd_limits[1]], bnd_limits, bnd_type
            )
        # For downwind
        else:
            unnt[:, i + 1] = np.pad(tmmp[bnd_limits[0]:], bnd_limits, bnd_type)

    return t, unnt


def evolv_Lax_uadv_burgers(
    xx,
    hh,
    nt,
    cfl_cut=0.98,
    ddx=lambda x, y: deriv_dnw(x, y),
    bnd_type="wrap",
    bnd_limits=[0, 1],
    **kwargs,
):
    r"""
    Advance nt time-steps in time the burger eq for a being u using the Lax
    method.

    Requires
    --------
    step_uadv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    cfl_cut : `array`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `array`
        Lambda function allows to change the space derivative function.
        By derault  lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    """
    N = np.size(xx)

    unnt = np.zeros((N, nt))
    unnt[:, 0] = hh

    t = np.zeros(nt)

    dx = xx[1] - xx[0]

    for i in range(0, nt - 1):
        #dx = np.gradient(xx)
        #dt = np.min(dx/np.abs(unnt[:, i]))*cfl_cut
        dt, tmp = step_adv_burgers(xx, hh=unnt[:, i], a=unnt[:, i], cfl_cut=cfl_cut, ddx=ddx)
        t[i + 1] = t[i] + dt
        term1 = np.roll(unnt[:, i], -1) + np.roll(unnt[:, i], 1)
        term2 = np.roll(unnt[:, i], -1) - np.roll(unnt[:, i], 1)
        frac = (unnt[:,i]*dt) / (2*dx)
        tmmp = 0.5*term1 - frac*term2 
        # For upwind and centre
        if bnd_limits[1] > 0:
            unnt[:, i + 1] = np.pad(
                tmmp[bnd_limits[0]: -bnd_limits[1]], bnd_limits, bnd_type
            )
        # For downwind
        else:
            unnt[:, i + 1] = np.pad(tmmp[bnd_limits[0]:], bnd_limits, bnd_type)

    return t, unnt


def evolv_Lax_adv_burgers(
    xx,
    hh,
    nt,
    a,
    cfl_cut=0.98,
    ddx=lambda x, y: deriv_dnw(x, y),
    bnd_type="wrap",
    bnd_limits=[0, 1],
    **kwargs,
):
    r"""
    Advance nt time-steps in time the burger eq for a being a a fix constant or
    array.

    Requires
    --------
    step_adv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    """


def step_uadv_burgers(xx, hh, cfl_cut=0.98, ddx=lambda x, y: deriv_dnw(x, y), **kwargs):
    r"""
    Right hand side of Burger's eq. where a is u, i.e hh.

    Requires
    --------
        cfl_adv_burger function which computes np.min(dx/a)

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    cfl_cut : `array`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function`
        Allows to select the type of spatial derivative.
        By default lambda x,y: deriv_dnw(x, y)


    Returns
    -------
    dt : `array`
        time interval
    unnt : `array`
        right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x}
    """

def evolv_Rie_uadv_burgers(xx, 
                           hh, 
                           nt, 
                           cfl_cut=0.98, 
                           bnd_type="wrap",
                           bnd_limits=[0, 1],
                           **kwargs):
    r"""
    Right hand side of Burger's eq. where a is u, i.e hh.

    Requires
    --------
        cfl_adv_burger function which computes np.min(dx/a)

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    cfl_cut : `array`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function`
        Allows to select the type of spatial derivative.
        By default lambda x,y: deriv_dnw(x, y)


    Returns
    -------
    dt : `array`
        time interval
    unnt : `array`
        right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x}
    """
    N = np.size(xx)

    unnt = np.zeros((N, nt))
    unnt[:, 0] = hh
    
    t = np.zeros(nt)
    
    for it in range(nt-1):
        # Get the left and right u
        u_L = unnt[:-1, it]
        u_R = unnt[1:, it]
        # fix boundaries
        u_L = np.pad(u_L, (bnd_limits[0], bnd_limits[1]),
                     bnd_type)
        u_R = np.pad(u_R, (bnd_limits[1], bnd_limits[0]),
                     bnd_type)
        
        # calcute the flux
        F_R = .5*u_R**2
        F_L = .5*u_L**2
        
        # Gets the velocities
        va = np.max([np.abs(u_R), np.abs(u_L)], axis=0)
        dx = np.gradient(xx)
        
        # Timestep and rhs
        dt = cfl_cut * cfl_adv_burger(va, xx)
        rhs = .5*(F_R + F_L) - .5*va*(u_R - u_L)

        # Calculates the next unnt
        unnt[:, it + 1] = unnt[:, it] - dt*(rhs - np.roll(rhs, 1))/dx
        t[it+1] = t[it] + dt
    return t, unnt

def cfl_diff_burger(a, x):
    r"""
    Computes the dt_fact, i.e., Courant, Fredrich, and Lewy condition for the
    diffusive term in the Burger's eq.

    Parameters
    ----------
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    x : `array`
        Spatial axis.

    Returns
    -------
    `float`
        min(dx/|a|)
    """
    dx = np.gradient(x)
    return np.min(dx/np.abs(a))


def ops_Lax_LL_Add(
    xx,
    hh,
    nt,
    a,
    b,
    cfl_cut=0.98,
    ddx=lambda x, y: deriv_cent(x, y),
    bnd_type="wrap",
    bnd_limits=[0, 1],
    **kwargs,
):
    r"""
    Advance nt time-steps in time the burger eq for a being a and b a fix
    constant or array. Solving two advective terms separately with the Additive
    Operator Splitting scheme.  Both steps are with a Lax method.

    Requires
    --------
    step_adv_burgers
    cfl_adv_burger

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    b : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries
        By default 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    """
    N = np.size(xx)
    
    t = np.zeros(nt)
    unnt = np.zeros((N, nt))

    unnt[:, 0] = hh
    
    for i in range(nt-1):
        dtu, du = step_adv_burgers(xx=xx, hh=unnt[:, i], 
                                  a=a, cfl_cut=cfl_cut, ddx=ddx)
        dtv, dv = step_adv_burgers(xx=xx, hh=unnt[:, i], 
                                  a=b, cfl_cut=cfl_cut, ddx=ddx)
        
        dt = np.min([dtu, dtv])

        # LAX method forwards
        u = .5 * (np.roll(unnt[:, i], -1) + np.roll(unnt[:, i], 1)) - du*dt
        v = .5 * (np.roll(unnt[:, i], -1) + np.roll(unnt[:, i], 1)) - dv*dt
        
        
        if bnd_limits[1] > 0:
            u = np.pad(
                u[bnd_limits[0]: -bnd_limits[1]], bnd_limits, bnd_type)
            v= np.pad(
                v[bnd_limits[0]: -bnd_limits[1]], bnd_limits, bnd_type)
        # For downwind
        else:
            u = np.pad(u[bnd_limits[0]:], bnd_limits, bnd_type)
            v = np.pad(v[bnd_limits[0]:], bnd_limits, bnd_type)
        
        unnt[:, i +  1] = u + v - unnt[:, i]
        t[i + 1] = t[i] + dt

    return t, unnt


def ops_Lax_LL_Lie(
    xx,
    hh,
    nt,
    a,
    b,
    cfl_cut=0.98,
    ddx=lambda x, y: deriv_cent(x, y),
    bnd_type="wrap",
    bnd_limits=[0, 1],
    **kwargs,
):
    r"""
    Advance nt time-steps in time the burger eq for a being a and b a fix
    constant or array. Solving two advective terms separately with the Lie-
    Trotter Operator Splitting scheme.  Both steps are with a Lax method.

    Requires:
    step_adv_burgers
    cfl_adv_burger

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    b : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    """
    N = np.size(xx)
    
    t = np.zeros(nt)
    unnt = np.zeros((N, nt))

    unnt[:, 0] = hh
    
    for i in range(nt-1):
        dtu, du = step_adv_burgers(xx=xx, hh=unnt[:, i], a=a, cfl_cut=cfl_cut, ddx=ddx)
        dt2 = cfl_adv_burger(a=b, x=xx)

        dt = np.min([dtu, dt2])

        # u forwards
        u = .5 * (np.roll(unnt[:, i], -1) + np.roll(unnt[:, i], 1)) - du*dt

        if bnd_limits[1] > 0:
            u = np.pad(
                u[bnd_limits[0]: -bnd_limits[1]], bnd_limits, bnd_type)
        # For downwind
        else:
            u = np.pad(u[bnd_limits[0]:], bnd_limits, bnd_type)

        # dv with u
        tmp, dv = step_adv_burgers(xx=xx, hh=u, a=b, cfl_cut=cfl_cut, ddx=ddx)

        # v forwards
        v = .5 * (np.roll(u, -1) + np.roll(u, 1)) - dv*dt
        
        if bnd_limits[1] > 0:
            v = np.pad(
                v[bnd_limits[0]: -bnd_limits[1]], bnd_limits, bnd_type)
        # For downwind
        else:
            v = np.pad(v[bnd_limits[0]:], bnd_limits, bnd_type)

        unnt[:, i + 1] = v
        t[i + 1] = t[i] + dt
    return t, unnt

def ops_Lax_LL_Strang(
    xx,
    hh,
    nt,
    a,
    b,
    cfl_cut=0.98,
    ddx=lambda x, y: deriv_cent(x, y),
    bnd_type="wrap",
    bnd_limits=[0, 1],
    **kwargs,
):
    r"""
    Advance nt time-steps in time the burger eq for a being a and b a fix
    constant or array. Solving two advective terms separately with the Lie-
    Trotter Operator Splitting scheme. Both steps are with a Lax method.

    Requires
    --------
    step_adv_burgers
    cfl_adv_burger
    numpy.pad for boundaries.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    b : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        Allows to select the type of boundaries.
        By default `wrap`
    bnd_limits : `list(int)`
        The number of pixels that will need to be updated with the boundary information.
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    """
    N = np.size(xx)
    
    t = np.zeros(nt)
    unnt = np.zeros((N, nt))

    unnt[:, 0] = hh
    
    for i in range(nt-1):
        dtu, du = step_adv_burgers(xx=xx, hh=unnt[:, i], a=a, cfl_cut=cfl_cut, ddx=ddx)
        dt2 = cfl_adv_burger(a=b, x=xx)

        dt = np.min([dtu, dt2])

        # u forwards
        u = .5 * (np.roll(unnt[:, i], -1) + np.roll(unnt[:, i], 1)) - du*dt/2

        if bnd_limits[1] > 0:
            u = np.pad(
                u[bnd_limits[0]: -bnd_limits[1]], bnd_limits, bnd_type)
        # For downwind
        else:
            u = np.pad(u[bnd_limits[0]:], bnd_limits, bnd_type)

        # dv with u
        tmp, dv = step_adv_burgers(xx=xx, hh=u, a=b, cfl_cut=cfl_cut, ddx=ddx)

        # v forwards
        v = .5 * (np.roll(u, -1) + np.roll(u, 1)) - dv*dt
        
        if bnd_limits[1] > 0:
            v = np.pad(
                v[bnd_limits[0]: -bnd_limits[1]], bnd_limits, bnd_type)
        # For downwind
        else:
            v = np.pad(v[bnd_limits[0]:], bnd_limits, bnd_type)

        tmp, dw = step_adv_burgers(xx=xx, hh=v, a=a, cfl_cut=cfl_cut, ddx=ddx)

        # w forwards
        w = .5 * (np.roll(v, -1) + np.roll(v, 1)) - dw*dt/2

        unnt[:, i + 1] = w
        t[i + 1] = t[i] + dt
    return t, unnt


def ops_Lax_LH_Strang(
    xx,
    hh,
    nt,
    a,
    b,
    cfl_cut=0.98,
    ddx=lambda x, y: deriv_cent(x, y),
    bnd_type="wrap",
    bnd_limits=[0, 1],
    **kwargs,
):
    r"""
    Advance nt time-steps in time the burger eq for a being a and b a fix
    constant or array. Solving two advective terms separately with the Strang
    Operator Splitting scheme. One step is with a Lax method and the second
    step is the Hyman predictor-corrector scheme.

    Requires
    --------
    step_adv_burgers
    cfl_adv_burger

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    b : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    """
    N = np.size(xx)
    
    t = np.zeros(nt)
    unnt = np.zeros((N, nt))

    unnt[:, 0] = hh
    
    for i in range(nt-1):
        dtu, du = step_adv_burgers(xx=xx, hh=unnt[:, i], a=a, cfl_cut=cfl_cut, ddx=ddx)
        dt2 = cfl_adv_burger(a=b, x=xx)

        dt = np.min([dtu, dt2])

        # u forwards
        u = .5 * (np.roll(unnt[:, i], -1) + np.roll(unnt[:, i], 1)) - du*dt/2

        if bnd_limits[1] > 0:
            u = np.pad(
                u[bnd_limits[0]: -bnd_limits[1]], bnd_limits, bnd_type)
        # For downwind
        else:
            u = np.pad(u[bnd_limits[0]:], bnd_limits, bnd_type)

        # dv with u
        tmp, dv = step_adv_burgers(xx=xx, hh=u, a=b, cfl_cut=cfl_cut, ddx=ddx)

        # v forwards
        v = .5 * (np.roll(u, -1) + np.roll(u, 1)) - dv*dt
        
        if (i == 0):
            v, u_old, dt_old = hyman(xx, u, dv, a=b, cfl_cut=cfl_cut, ddx=ddx, bnd_limits=bnd_limits)
        else:
            v, dv, dt_old = hyman(xx, u, dv, a=b, fold=u_old, dtold=dt_old,
                              cfl_cut=cfl_cut, ddx=ddx, bnd_limits=bnd_limits)
        
        if bnd_limits[1] > 0:
            v = np.pad(
                v[bnd_limits[0]: -bnd_limits[1]], bnd_limits, bnd_type)
        # For downwind
        else:
            v = np.pad(v[bnd_limits[0]:], bnd_limits, bnd_type)

        tmp, dw = step_adv_burgers(xx=xx, hh=v, a=a, cfl_cut=cfl_cut, ddx=ddx)

        # w forwards
        w = .5 * (np.roll(v, -1) + np.roll(v, 1)) - dw*dt/2

        unnt[:, i + 1] = w
        t[i + 1] = t[i] + dt
    return t, unnt


def step_diff_burgers(xx, hh, a, ddx=lambda x, y: deriv_cent(x, y), **kwargs):
    r"""
    Right hand side of the diffusive term of Burger's eq. where nu can be a
    constant or a function that depends on xx.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_dnw(x, y)

    Returns
    -------
    `array`
        Right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x}
    """
    dt = cfl_diff_burger(a, xx)
    ans = ddx(xx, hh) - hh
    return ans/dt
    


def NR_f(xx, un, uo, a, dt, **kwargs):
    r"""
    NR F function.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    un : `array`
        Function that depends on xx.
    uo : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `float`
        Time interval

    Returns
    -------
    `array`
        function  u^{n+1}_{j}-u^{n}_{j} - a (u^{n+1}_{j+1} - 2 u^{n+1}_{j} -u^{n+1}_{j-1}) dt
    """
    F = un - uo - a*(np.roll(un, -1) - 2*un - np.roll(un, 1)) * dt
    return F


def jacobian(xx, un, a, dt, **kwargs):
    r"""
    Jacobian of the F function.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    un : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `float`
        Time interval

    Returns
    -------
    `array`
        Jacobian F_j'(u^{n+1}{k})
    """
    dx = xx[1] - xx[0]
    dx2 = dx*dx
    N = np.size(xx)
    F = np.zeros((N, N))

    for x in range(N):
        term = (dt*a)/dx2
        F[x, x] = 1 + 2*term
        if x < N - 1:
            F[x, x+1] = -term
        if x > 1:
            F[x, x-1] = -term
    return F


def Newton_Raphson(
    xx, hh, a, dt, nt, toll=1e-5, ncount=2, bnd_type="wrap", bnd_limits=[1, 1], **kwargs
):
    r"""
    NR scheme for the burgers equation.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `float`
        Time interval
    nt : `int`
        Number of iterations
    toll : `float`
        Error limit.
        By default 1e-5
    ncount : `int`
        Maximum number of iterations.
        By default 2
    bnd_type : `string`
        Allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [1,1]

    Returns
    -------
    t : `array`
        Array of time.
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    errt : `array`
        Error for each timestep
    countt : `list(int)`
        number iterations for each timestep
    """
    err = 1.0
    unnt = np.zeros((np.size(xx), nt))
    errt = np.zeros((nt))
    countt = np.zeros((nt))
    unnt[:, 0] = hh
    t = np.zeros((nt))

    # Looping over time
    for it in range(1, nt):
        uo = unnt[:, it - 1]
        ug = unnt[:, it - 1]
        count = 0
        # iteration to reduce the error.
        while (err >= toll) and (count < ncount):

            jac = jacobian(xx, ug, a, dt)  # Jacobian
            ff1 = NR_f(xx, ug, uo, a, dt)  # F
            # Inversion:
            un = ug - np.matmul(np.linalg.inv(jac), ff1)

            # error:
            err = np.max(np.abs(un - ug) / (np.abs(un) + toll))  # error
            # err = np.max(np.abs(un-ug))
            errt[it] = err

            # Number of iterations
            count += 1
            countt[it] = count

            # Boundaries
            if bnd_limits[1] > 0:
                u1_c = un[bnd_limits[0]: -bnd_limits[1]]
            else:
                u1_c = un[bnd_limits[0]:]
            un = np.pad(u1_c, bnd_limits, bnd_type)
            ug = un
        err = 1.0
        t[it] = t[it - 1] + dt
        unnt[:, it] = un

    return t, unnt, errt, countt


def NR_f_u(xx, un, uo, dt, **kwargs):
    r"""
    NR F function.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    un : `array`
        Function that depends on xx.
    uo : `array`
        Function that depends on xx.
    a : `float` and `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `int`
        Time interval

    Returns
    -------
    `array`
        function  u^{n+1}_{j}-u^{n}_{j} - a (u^{n+1}_{j+1} - 2 u^{n+1}_{j} -u^{n+1}_{j-1}) dt
    """


def jacobian_u(xx, un, dt, **kwargs):
    """
    Jacobian of the F function.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    un : `array`
        Function that depends on xx.
    a : `float` and `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `int`
        Time interval

    Returns
    -------
    `array`
        Jacobian F_j'(u^{n+1}{k})
    """


def Newton_Raphson_u(
    xx, hh, dt, nt, toll=1e-5, ncount=2, bnd_type="wrap", bnd_limits=[1, 1], **kwargs
):
    """
    NR scheme for the burgers equation.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    dt : `float`
        Time interval
    nt : `int`
        Number of iterations
    toll : `float`
        Error limit.
        By default 1-5
    ncount : `int`
        Maximum number of iterations.
        By default 2
    bnd_type : `string`
        Allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [1,1]

    Returns
    -------
    t : `array`
        Time.
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    errt : `array`
        Error for each timestep
    countt : `array(int)`
        Number iterations for each timestep
    """
    err = 1.0
    unnt = np.zeros((np.size(xx), nt))
    errt = np.zeros((nt))
    countt = np.zeros((nt))
    unnt[:, 0] = hh
    t = np.zeros((nt))

    # Looping over time
    for it in range(1, nt):
        uo = unnt[:, it - 1]
        ug = unnt[:, it - 1]
        count = 0
        # iteration to reduce the error.
        while (err >= toll) and (count < ncount):

            jac = jacobian_u(xx, ug, dt)  # Jacobian
            ff1 = NR_f_u(xx, ug, uo, dt)  # F
            # Inversion:
            un = ug - np.matmul(np.linalg.inv(jac), ff1)

            # error
            err = np.max(np.abs(un - ug) / (np.abs(un) + toll))
            errt[it] = err

            # Number of iterations
            count += 1
            countt[it] = count

            # Boundaries
            if bnd_limits[1] > 0:
                u1_c = un[bnd_limits[0]: -bnd_limits[1]]
            else:
                u1_c = un[bnd_limits[0]:]
            un = np.pad(u1_c, bnd_limits, bnd_type)
            ug = un
        err = 1.0
        t[it] = t[it - 1] + dt
        unnt[:, it] = un

    return t, unnt, errt, countt


def taui_sts(nu, niter, iiter):
    """
    STS parabolic scheme. [(nu -1)cos(pi (2 iiter - 1) / 2 niter) + nu + 1]^{-1}

    Parameters
    ----------
    nu : `float`
        Coefficient, between (0,1).
    niter : `int`
        Number of iterations
    iiter : `int`
        Iterations number

    Returns
    -------
    `float`
        [(nu -1)cos(pi (2 iiter - 1) / 2 niter) + nu + 1]^{-1}
    """


def evol_sts(
    xx,
    hh,
    nt,
    a,
    cfl_cut=0.45,
    ddx=lambda x, y: deriv_cent(x, y),
    bnd_type="wrap",
    bnd_limits=[0, 1],
    nu=0.9,
    n_sts=10,
):
    """
    Evolution of the STS method.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.45
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_cent(x, y)
    bnd_type : `string`
        Allows to select the type of boundaries
        by default 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By defalt [0,1]
    nu : `float`
        STS nu coefficient between (0,1).
        By default 0.9
    n_sts : `int`
        Number of STS sub iterations.
        By default 10

    Returns
    -------
    t : `array`
        time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    """


def hyman(
    xx,
    f,
    dth,
    a,
    fold=None,
    dtold=None,
    cfl_cut=0.8,
    ddx=lambda x, y: deriv_dnw(x, y),
    bnd_type="wrap",
    bnd_limits=[0, 1],
    **kwargs,
):

    dt, u1_temp = step_adv_burgers(xx, f, a, ddx=ddx)

    if np.any(fold) == None:
        fold = np.copy(f)
        f = (np.roll(f, 1) + np.roll(f, -1)) / 2.0 + u1_temp * dth
        dtold = dth

    else:
        ratio = dth / dtold
        a1 = ratio**2
        b1 = dth * (1.0 + ratio)
        a2 = 2.0 * (1.0 + ratio) / (2.0 + 3.0 * ratio)
        b2 = dth * (1.0 + ratio**2) / (2.0 + 3.0 * ratio)
        c2 = dth * (1.0 + ratio) / (2.0 + 3.0 * ratio)

        f, fold, fsav = hyman_pred(f, fold, u1_temp, a1, b1, a2, b2)

        if bnd_limits[1] > 0:
            u1_c = f[bnd_limits[0]: -bnd_limits[1]]
        else:
            u1_c = f[bnd_limits[0]:]
        f = np.pad(u1_c, bnd_limits, bnd_type)

        dt, u1_temp = step_adv_burgers(xx, f, a, cfl_cut, ddx=ddx)

        f = hyman_corr(f, fsav, u1_temp, c2)

    if bnd_limits[1] > 0:
        u1_c = f[bnd_limits[0]: -bnd_limits[1]]
    else:
        u1_c = f[bnd_limits[0]:]
    f = np.pad(u1_c, bnd_limits, bnd_type)

    dtold = dth

    return f, fold, dtold


def hyman_corr(f, fsav, dfdt, c2):
    return fsav + c2 * dfdt


def hyman_pred(f, fold, dfdt, a1, b1, a2, b2):
    fsav = np.copy(f)
    tempvar = f + a1 * (fold - f) + b1 * dfdt
    fold = np.copy(fsav)
    fsav = tempvar + a2 * (fsav - tempvar) + b2 * dfdt
    f = tempvar

    return f, fold, fsav


def animation(xx, ut, U, nt, t, nrows=1, ncols=1, figsize=(10, 5),
                label_1="Numerical", label_2="Analytical"):
    """
    Animates the functions.

    Parameters
    ----------
    xx  :   `array`
         Spatial array
    ut  :   `array`
         Numerical solution array
    U   :   `array`
         Analytical solution array
    nt  :   `float` or `int`
         number of time points
    nrows : `int`
         number of rows
    ncols : `int`
         number of cols 
    figsize : `tuple`
         The figure size

    Returns
    -------
    Animation
    """
    modulename = "matplotlib"
    if modulename not in dir():
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    def init():
        axes.plot(xx, ut[:, 0], label=label_1)
        axes.plot(xx, U[:, 0], label=label_2)
        axes.legend()

    def animate(i):

        axes.clear()
        axes.plot(xx, ut[:, i], label=label_1)
        axes.plot(xx, U[:, i], label=label_2)
        axes.legend()
        axes.set_title("t=%.2f" % t[i])

    anim = FuncAnimation(fig, animate, interval=1, frames=nt, init_func=init)
    return anim
