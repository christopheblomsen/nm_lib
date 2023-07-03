#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 02 10:25:17 2021.

@Supervisor: Juan Martinez Sykora
@author: Christophe Kristian Blomsen
"""

# import builtin modules

# import external public "common" modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import fsolve


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
    y = ((np.roll(hh, -1) - np.roll(hh, 0)) 
         / (np.roll(xx, -1) - np.roll(xx, 0)))
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
        min(dx**2/2|a|)
    """
    dx = np.gradient(x)
    return np.min(dx**2/(2*np.abs(a)))


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
        
        if(i == 0):
            v, u_old, dt_old = hyman(xx, v, dt, a=b, cfl_cut=cfl_cut, ddx=ddx, bnd_limits=bnd_limits)
        else:
            v, u_old, dt_old = hyman(xx, v, dt, a=b, fold=u_old, dtold=dt_old,
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


def step_diff_burgers(xx, hh, a, cfl_cut, ddx=lambda x, y: deriv_cent(x, y), **kwargs):
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
    eps = 1e-10
    dt = cfl_cut*cfl_diff_burger(a, xx)
    rhs = a*ddx(xx, ddx(xx, hh))
    return dt, rhs
    
def evolv_diff_burgers(xx, hh, nt, a, ddx = lambda x, y: deriv_cent(x, y),
                       cfl_cut=0.98, bnd_type='wrap', bnd_limits=[0,1]):

    N = np.size(xx)
    
    t = np.zeros(nt)
    unnt = np.zeros((N, nt))
    
    unnt[:, 0] = hh

    for i in range(0, nt-1):
        dt, rhs = step_diff_burgers(xx, unnt[:, i], a, ddx=ddx, cfl_cut=cfl_cut)
        u = unnt[:, i] + rhs*dt
        
        if bnd_limits[1] > 0:
            u = np.pad(
                u[bnd_limits[0]: -bnd_limits[1]], bnd_limits, bnd_type)
        # For downwind
        else:
            u = np.pad(u[bnd_limits[0]:], bnd_limits, bnd_type)

        unnt[:, i+1] = hh
        t[i+1] = t[i] + dt
    return t, unnt

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
        function  u^{n+1}_{j}-u^{n}_{j} - a (u^{n+1}_{j+1} - 2 u^{n+1}_{j} +u^{n+1}_{j-1}) dt
    """
    dx = np.gradient(xx)
    F = un - uo - a*(np.roll(un, -1) - 2*un + np.roll(un, 1)) * (dt/dx**2)
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

    """
    np fill diagonal. upper [1:] -c, lower [:, 1:] -c
    diag J, 1 + 2*C
    """
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
        function  u^{n+1}_{j}-u^{n}_{j} - a (u^{n+1}_{j+1} - 2 u^{n+1}_{j} +u^{n+1}_{j-1}) dt
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
    arg = np.pi*(2*iiter - 1) / (2*niter)
    # ans = 1./((nu - 1)*np.cos(arg) + nu + 1)
    ans = ((nu - 1)*np.cos(arg) + nu + 1)**(-1)
    return ans

def tau_sts(nu, n, dt_cfl):
    """
    Uses the super time stepping method
    Calculates the dt_sts

    Parameters:
    -----------
    nu  :  `float`
            dampening parameter
    n   :   `int`
            number of smaller steps
    dt  :   `float`
            cfl condition
    
    Returns:
    --------
    dt_sts  :   `float`
            the super time stepping dt
    """
    a = n/(2*np.sqrt(nu))
    b = (1 + np.sqrt(nu))**2*n - (1 - np.sqrt(nu))**2*n
    c = (1 + np.sqrt(nu))**2*n + (1 - np.sqrt(nu))**2*n
    
    dt_sts = a*(b/c)*dt_cfl
    return dt_sts

def evolv_sts(
    xx,
    hh,
    nt,
    a,
    cfl_cut=0.98,
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
    N = np.size(xx)
    unnt = np.zeros((N, nt))
    unnt[:, 0] = hh
    t = np.zeros(nt)
    tsts = []
    dx = xx[1] - xx[0]
    tcfl = (dx**2)/a
    
    for n in range(nt-1):
        ts = []
        unts = np.zeros((N, n_sts))
        unts[:, 0] = unnt[:, n]
        for it in range(0, n_sts-1):
            dti = tcfl*taui_sts(nu, n_sts, it)
            dt, u1_temp = step_diff_burgers(xx, unts[:, it], a,
                                            cfl_cut=cfl_cut, ddx=ddx)
            u1_temp = unts[:, it] + u1_temp*dti
            # Boundaries
            if bnd_limits[1] > 0:
                u1_c = u1_temp[bnd_limits[0]: -bnd_limits[1]]
            else:
                u1_c = u1_temp[bnd_limits[0]:]
            unts[:, it+1] = np.pad(u1_c, bnd_limits, bnd_type)
            unts[:, it+1] = u1_temp
            unntmp = unts[:, it+1]
            ts.append(dti)
        tsts.append(ts)
        unnt[:, n+1] = unntmp
        t[n+1] = t[n] + np.sum(ts)
    return t, unnt, tsts


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

""" 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++
SOD TUBE below
"""

def shock_function(P_prime, γ=5/3,
                   rhoL=1., rhoR=0.125,
                    PL=1., PR=0.1):
    """
    Shock function to calculate P' = P1 for when u1 = u2

    Parameters
    ----------
    P_prime :   `float`
                The initial guess of the Pressure
    γ       :   `float`
                Default 5/3
    rhoL    :   `float`
                Initial density left side
                Default 1.
    rhoR    :   `float`
                Initial density right side
                Default 0.125
    PL      :   `float`
                Initial Pressure left side
                Default 1.
    PR      :   `float`
                Initial Pressure right side
                Default 0.1
                
    Returns
    ---------
    P_prime : `float`
    """
    Γ = (γ - 1)/(γ + 1)
    β = (γ - 1)/(2*γ)
    u1 = (P_prime - PR)*np.sqrt((1 - Γ)/(rhoR*(P_prime + Γ*PR)))
    
    u2 = (PL**β - P_prime**β)*np.sqrt((1 - Γ**2)*PL**(1/γ)/Γ**2*rhoL)
    
    return u1 - u2

def iterate(function, guess):
    """
    Used to interativly solve for the Pressure in the expansion region
    
    Parameters
    ----------
    function    : `function`
                    By design should be `shock_function`
    guess       : `float`
                    The initial guess
    
    Returns
    ---------
    Root :  `float`
            root of the function used to determined P'
    """
    ans = fsolve(function, guess)
    return ans[0]

def shock_region(P_prime, Γ,
            rhoR=0.125, PR=0.1):
    """
    Calculates the profiles in the shock region

    Parameters:
    -----------
    P_prime : `float`
                The pressure
    Γ       : `float`
                Coefficient dependent on adiabatic gamma

    Returns:
    ----------
    profiles    : `list`
                list of all the profiles within the region 
    """
    
    P1 = P_prime
    rho1 = rhoR*(P1 + Γ*PR)/(PR + Γ*P1)
    u1 = (P_prime - PR)*np.sqrt((1 - Γ)/(rhoR*(P_prime + Γ*PR)))
    
    return P1, rho1, u1

def contact_region(P1, u1,
            PL=1.,
            rhoL=1., γ=5/3):
    """
    Calculates the profiles in the contact region 

    Parameters:
    -----------
    P1      : `float`
                The pressure from region 1
    u1      : `float`
                velocity from region 1
    PL      : `float`
                The left boundary pressure
                Default 1.
    rhoL    : `float`
                The left boundary density
                Default 1.
    γ       : `float`
                Adiabatic gamma
                Default 5/3

    Returns:
    ----------
    profiles    : `list`
                list of all the profiles within the region 
    """
    u2 = u1
    P2 = P1
    rho2 = rhoL*(P2/PL)**(1/γ)
    return P2, rho2, u2

def expansion_fan_region(xx, tt, c1, 
                        γ=5/3, PL=1., x0=0.5):
    """
    Calculates the profiles in the expansion fan region 

    Parameters:
    -----------
    xx      : `array`
                Spatial domain of test
    tt      : `float`
                Time point of calculation
    γ       : `float`
                Adiabatic gamma
                Default 5/3
    PL      : `float`
                The left boundary pressure
                Default 1.
    x0    : `float`
                Middle of the spatial domain
                Default 0.5

    Returns:
    ----------
    profiles    : `list`
                list of all the profiles within the region 
    """
    u_e = 2/(γ + 1)*(c1 + (xx - x0)/tt)
    
    # Varying sound speed over region
    c_e = c1 - (γ - 1)*u_e/2
    
    P_e = PL*(c_e/c1)**(2*γ/(γ - 1))
    
    rho_e = γ*P_e/c_e**2
    
    return P_e, rho_e, u_e

def intersect_left2expansion(tt, c1, x0=0.5):
    """
    Calculates the intersect between left boundary
    region and the expansion fan region

    Parameters
    ----------
    tt  :   `float`
            Time point of the calculation
    c1  :   `float`
            Sound speed from left boundary
    x0  :   `float`
            Middle of spatial domain
            Default 0.5
    
    Returns:
    ---------
    intersect_point :   `float`
                        intersect point between regions
    
    """
    return x0 - c1*tt

def intersect_expansion2contact(tt, u2, c2, x0=0.5):
    """
    Calculates the intersect between expansion fan region
    and the contact region

    Parameters
    ----------
    tt  :   `float`
            Time point of the calculation
    u2  :   `float`
            Velocity from the contact region
    c2  :   `float`
            Sound speed from the contact region
    x0  :   `float`
            Middle of spatial domain
            Default 0.5
    
    Returns:
    ---------
    intersect_point :   `float`
                        intersect point between regions
    
    """
    return x0 + (u2 - c2)*tt

def intersect_contact2shock(tt, u2, x0=0.5):
    """
    Calculates the intersect between contact region
    and the shock region

    Parameters
    ----------
    tt  :   `float`
            Time point of the calculation
    u2  :   `float`
            Velocity from the contact region
    x0  :   `float`
            Middle of spatial domain
            Default 0.5
    
    Returns:
    ---------
    intersect_point :   `float`
                        intersect point between regions
    
    """
    return x0 + u2*tt

def intersect_shock2right(tt, w, x0=0.5):
    """
    Calculates the intersect between shock region
    and the right boundary

    Parameters
    ----------
    tt  :   `float`
            Time point of the calculation
    w   :   `float`
            Scale factor for the intersect point
    x0  :   `float`
            Middle of spatial domain
            Default 0.5
    
    Returns:
    ---------
    intersect_point :   `float`
                        intersect point between regions
    
    """
    return x0 + w*tt

def sound_speed(P, rho, γ=5/3):
    """
    Calculates the sound speed

    Parameters
    ----------
    P   :   `float`
            Pressure
    rho :   `float`
            Density
    γ   :   Adiabatic gamma
            Default 5/3
    """
    return np.sqrt(γ*P/rho)

def make_array(qL, q_e, q2, q1, qR, N):
    """
    Makes an array out from floats

    Parameters
    ----------
    qL  :   `float`
            Left boundary value
    q_e  :   `array`
            expansion fan values
    q2  :   `float`
            contact region value
    q1  :   `float`
            shock region value
    qR  :   `float`
            Right boundary value
    N   :   `int`
            Sime of the regions
    """
    q = np.concatenate((qL*np.ones(N),
                        q_e,
                        q2*np.ones(N),
                        q1*np.ones(N),
                        qR*np.ones(N)))
    return q

def solve_sod_analytical(tt, x0=0.5, N=100, γ=5/3, 
                         rhoL=1., rhoR=0.125,
                         PL=1., PR=0.1,
                         uL=0., uR=0., 
                         eps=1e-10):
    """
    Solves the sod tube problem analytically

    Requiers
    ------------
    sound_speed
    iterate
    shock_function
    intersect_left2expansion
    intersect_expansion2contact
    intersect_contact2shock
    expansion_fan_region
    make_array

    Parameters
    ----------
    tt  :   `float`
            Time point of the calculation
    x0  :   `float`
            Middle of spatial domain
            Default 0.5
    N   :   `int`
            Number of points per domain
    γ   : `float`
            Adiabatic gamma
            Default 5/3
    rhoL: `float`
            The left boundary density
            Default 1.
    rhoR: `float`
            The right boundary density
            Default 0.125.
    PL  : `float`
            The left boundary pressure
            Default 1.
    PR  : `float`
            The right boundary pressure
            Default 0.1.
    uL  : `float`
            The left boundary velocity
            Default 0. 
    uR  : `float`
            The right boundary velocity
            Default 0..

    Returns
    ----------
    list    : `arrays`
                arrays of the profiles over the domain
    list[0] : `array`
                xx, spatial domain
    list[1] : `array`
                P, pressure over domain
    list[2] : `array`
                rho, density over domain
    list[3] : `array`
                velocity, density over domain
                
    """   

    c1 = sound_speed(PL, rhoL, γ=γ)
    c5 = sound_speed(PR, rhoR, γ=γ)

    Γ = (γ - 1)/(γ + 1)
    β = (γ - 1)/(2*γ)

    P_guess = (PL + PR)/2
    P_prime = iterate(shock_function, P_guess)
    
    P1, rho1, u1 = shock_region(P_prime, Γ, rhoR=rhoR, PR=PR)
    
    P2, rho2, u2 = contact_region(P1, u1, rhoL=rhoL, γ=γ)
    
    # find boundaries
    left2e = intersect_left2expansion(tt, c1, x0=x0)
    
    c2 = sound_speed(P2, rho2, γ=γ)
    
    # find boundaries around the contact region
    e2contact = intersect_expansion2contact(tt, u2, c2, x0=x0)
    contact2shock = intersect_contact2shock(tt, u2, x0=x0)
    
    z = P2/PR - 1

    fact = np.sqrt(1 + (γ + 1)/(2*γ)*z)
    
    shock2right = intersect_shock2right(tt, c5*fact, x0=x0)
    
    expansion_cell = np.linspace(left2e, e2contact, N)
    P_e, rho_e, u_e = expansion_fan_region(expansion_cell, tt, c1, γ=γ, PL=PL, x0=x0)
    
    # make the arrays of the domain
    P = make_array(PL, P_e, P2, P1, PR, N)
    rho = make_array(rhoL, rho_e, rho2, rho1, rhoR, N)
    u = make_array(uL, u_e, u2, u1, uR, N)
    
    xx = np.concatenate((np.linspace(0, left2e, N),
                         np.linspace(left2e, e2contact, N),
                         np.linspace(e2contact, contact2shock, N),
                         np.linspace(contact2shock, shock2right, N),
                         np.linspace(shock2right, 1, N)))
    
    return xx, P, rho, u

    
def init_sod_analytical(N=100, 
                        rhoL=1., rhoR=0.125,
                        PL=1., PR=0.1,
                        uL=0., uR=0.):
    """
    Initialises the domain for the sod analytical solution
    
    Parameters
    ----------
    N   :   `int`
            Number of points per domain
            Default 100
    rhoL: `float`
            The left boundary density
            Default 1.
    rhoR: `float`
            The right boundary density
            Default 0.125.
    PL  : `float`
            The left boundary pressure
            Default 1.
    PR  : `float`
            The right boundary pressure
            Default 0.1.
    uL  : `float`
            The left boundary velocity
            Default 0. 
    uR  : `float`
            The right boundary velocity
            Default 0..

    Returns
    ----------
    list    : `arrays`
                arrays of the profiles over the domain
    list[0] : `array`
                xx, spatial domain
    list[1] : `array`
                P, pressure over domain
    list[2] : `array`
                rho, density over domain
    list[3] : `array`
                velocity, density over domain
                
    """
    nx = int(5*N)
    nx2 = int(nx/2)

    x0 = spatial_domain(nx, x0=0, xf=1)
    P0 = np.ones(nx)
    rho0 = np.ones(nx)
    u0 = np.ones(nx)
    
    P0[:nx2] = PL
    P0[nx2:] = PR
    
    rho0[:nx2] = rhoL
    rho0[nx2:] = rhoR
    
    u0[:nx2] = uL
    u0[nx2:] = uR
    
    return x0, P0, rho0, u0

def solve_sod_array(t_end, nt, N=100, 
                    rhoL=1., rhoR=0.125,
                    PL=1., PR=0.1,
                    uL=0., uR=0.):
    """
    Solves the sod analytical solution to a time end
    over nt timesteps
    
    Parameters
    ----------
    t_end:  `float`
            Final timepoint
    nt  :   `int`
            Number of timepoints
    N   :   `int`
            Number of points per domain
            Default 100
    rhoL: `float`
            The left boundary density
            Default 1.
    rhoR: `float`
            The right boundary density
            Default 0.125.
    PL  : `float`
            The left boundary pressure
            Default 1.
    PR  : `float`
            The right boundary pressure
            Default 0.1.
    uL  : `float`
            The left boundary velocity
            Default 0. 
    uR  : `float`
            The right boundary velocity
            Default 0..

    Returns
    ----------
    list    : `arrays`
                arrays of the profiles over the domain
    list[0] : `array`
                xx, spatial domain
    list[1] : `array`
                P, pressure over domain
    list[2] : `array`
                rho, density over domain
    list[3] : `array`
                velocity, density over domain
    list[4] : `array`
                time, corresponding time array
    """
    
    t = np.linspace(0, t_end, nt)
    nx = int(5*N)
    x0, P0, rho0, u0 = init_sod_analytical(N, rhoL, rhoR, PL, PR, uL, uR)
    
    x = np.zeros((nx, nt))
    P = np.zeros((nx, nt))
    rho = np.zeros((nx, nt))
    u = np.zeros((nx, nt))
    
    x[:, 0] = x0
    P[:, 0] = P0
    rho[:, 0] = rho0
    u[:, 0] = u0
    
    for it in range(1, nt):
        x_temp, P_temp, rho_temp, u_temp = solve_sod_analytical(t[it], N=N)
        x[:, it] = x_temp
        P[:, it] = P_temp
        rho[:, it] = rho_temp
        u[:, it] = u_temp
    
    return x, P, rho, u, t
        

def spatial_domain(nump, x0=-2.6, xf=2.6):
    """
    Gets the spatial domain

    Parameters
    ----------
    nump    :   `int`
                number of spatial points
    x0      :   `float`
                starting position
    xf      :    `float`
                final position
    
    Returns
    ---------
    xx      :   `array`
                spatial domain
    """
    if nump == 1:
        ans = np.array([0.])
    else:
        ans = np.arange(nump)/(nump - 1) * (xf - x0) + x0
    return ans

def animate_sod(x, P, rho, u, t, title):
    """
    Animates the sod solution
    """
    nt = len(t)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    def init():
        axes[0].plot(x[:, 0], P[:, 0], label='P', color='green')
        axes[1].plot(x[:, 0], rho[:, 0], label='rho', color='blue')
        axes[2].plot(x[:, 0], u[:, 0], label='u', color='red')
        for k in range(3):
            axes[k].set_title(f't=0 s')
            axes[k].legend()
        
    def animate(i):
        for k in range(3):
            axes[k].clear()
        axes[0].plot(x[:, i], P[:, i], label='P', color='green')
        axes[1].plot(x[:, i], rho[:, i], label='rho', color='blue')
        axes[2].plot(x[:, i], u[:, i], label='u', color='red')
        for k in range(3):
            axes[k].legend()
            axes[k].set_title(f't={t[i]:.2f} s')        
    
    anim = FuncAnimation(fig, animate, interval=10, frames=(nt-1), init_func=init)
    anim.save(f'{title}.mp4', writer='ffmpeg')

"""
Sod above
+++++++++++++++++++++++++++++++++++++
solver below
"""

def calculate_dt(xx, yy, zz, ux, uy, uz, cs, eps, nx, ny, nz, debug=False):
    """
    Calculates the dt according
    """
    u = np.sqrt(ux*ux + uy*uy + uz*uz)
    ans = []
    if nx > 1:
        ans.append(np.min(np.gradient(xx)) / np.max((np.abs(ux) + cs + eps)))
    if ny > 1:
        ans.append(np.min(np.gradient(yy)) / np.max((np.abs(uy) + cs + eps)))
    if nz > 1:
        ans.append(np.min(np.gradient(zz)) / np.max((np.abs(uz) + cs + eps)))
    res = np.min(ans)
    if debug:
        print(f'max u : {np.max(u)}')
        print(f'max cs : {np.max(cs)}')
    return res

def find_nan(arr, title, i):
    """
    Looks for nan values and breaks the loop
    """
    if np.any(np.isnan(arr)):
        print(f'{title} is nan at i: {i}')
        
def init_array(domain, nt, P0, rho0, ux0, uy0, uz0, e0, γ=5/3):
    """
    Initialises arrays in 4D in this style (x domain, y domain, z domain, time)
    
    Parameters
    ----------
    domain : `tuple`
                Spatial domain
    nt     :  `int`
                timepoints
    P0      : `array`
                Initial pressure array
    rho0    : `array`
                Initial pressure array
    ux0     : `array`
                Initial velocity x array
    uy0     : `array`
                Initial velocity y array
    uz0     : `array`
                Initial velocity z array
    e0      : `array`
                Initial pressure array
    γ       : `float`
                Adiabatic gamma
                Default 5/3
    
    Returns
    -------
    Initial conditions
    """
    idx = (slice(None), slice(None), slice(None), 0)

    e = np.zeros((*domain, nt))
    Pg = np.zeros((*domain, nt))
    momentx = np.zeros((*domain, nt))
    momenty = np.zeros((*domain, nt))
    momentz = np.zeros((*domain, nt))
    rho = np.zeros((*domain, nt))
    ux = np.zeros((*domain, nt))
    uy = np.zeros((*domain, nt))
    uz = np.zeros((*domain, nt))

    e[idx] = e0
    Pg[idx] = P0
    rho[idx] = rho0
    ux[idx] = ux0
    uy[idx] = uy0
    uz[idx] = uz0
    momentx[idx] = ux0*rho0
    momenty[idx] = uy0*rho0
    momentz[idx] = uz0*rho0

    return Pg, e, momentx, momenty, momentz, rho, ux, uy, uz

def solver(domain, xx, yy, zz, nt, P0, rho0, ux0, uy0, uz0, e0,
           γ=5/3, cfl_cut=0.2, ddx=lambda x, y: deriv_cent(x, y),
           method='FTCS', debug=False):
    """
    Solves the EOS
    
    Parameters
    ----------
    domain : `tuple`
                Spatial domain
    xx      : `array`
                x spatial domain
    yy      : `array`
                y spatial domain
    zz      : `array`
                z spatial domain
    nt     :  `int`
                timepoints
    P0      : `array`
                Initial pressure array
    rho0    : `array`
                Initial pressure array
    ux0     : `array`
                Initial velocity x array
    uy0     : `array`
                Initial velocity y array
    uz0     : `array`
                Initial velocity z array
    e0      : `array`
                Initial pressure array
    γ       : `float`
                Adiabatic gamma
                Default 5/3
    cfl_cut : `float`
                Default 0.2
    ddx     : `function`
                The numerical scheme for derivation
                Default central scheme
    method  : `string`
                The numerical scheme for solving EOS
                Default FTCS
    debug   : `bool`
                Used to get debug print statement
                Default False
    
    Returns
    -------
    Arrays of the state for

    Pg, rho, momentx, momenty, momentz, e and corresponding time
    """
    nx, ny, nz = domain
    Pg, e, momentx, momenty, momentz, rho, ux, uy, uz = init_array(domain, nt, P0, rho0, ux0, uy0, uz0, e0, γ=γ)
    time = np.zeros(nt)
    eps = 1.e-10
    
    for it in range(0, nt-1):
        idx = (slice(None), slice(None), slice(None), it)
        idx_next = (slice(None), slice(None), slice(None), it+1)
        arg = np.abs(γ*Pg[idx] / (rho[idx] + eps))
        cs = np.sqrt(arg)
        find_nan(cs, 'cs', it)
        find_nan(rho[idx], 'rho', it)
        find_nan(Pg[idx], 'Pg', it)
        if debug:
            print('arg')
            print(arg)
            print('cs')
            print(cs)
            print(f'arg value with corresponding cs nan : {arg[np.where(np.isnan(cs) )]}')
            print(f'Pg value with corresponding cs nan : {Pg[idx][np.where(np.isnan(cs) )]}')
            print(f'rho value with corresponding cs nan : {rho[idx][np.where(np.isnan(cs) )]}')

        ux[idx] = momentx[idx]/(rho[idx] + eps)
        uy[idx] = momenty[idx]/(rho[idx] + eps)
        uz[idx] = momentz[idx]/(rho[idx] + eps)

        # cfl condition
        dt = calculate_dt(xx, yy, zz, ux[idx], uy[idx], uz[idx], cs, eps, nx, ny, nz)
        dt *= cfl_cut
        if dt < 0:
            print(f'dt negative: {it}')
            break
        find_nan(dt, 'dt', it)
        if np.any(np.isnan(dt)):
            return Pg, rho, momentx, momenty, momentz, e, time
        if debug:
            print(f'ux: {ux.shape}')
            print(f'uy: {uy.shape}')
            print(f'uz: {uz.shape}')
            print(f'yy: {yy.shape}')
            print(f'rho: {rho[0, :, 0, i].shape}')
            print(f'cs: {cs.shape}')
            print(f'dt: {dt}')
            print(f'min arg : {np.min(arg)}')
            print(f'min cs : {np.min(cs)}')
            print(f'rho[idx_x] shape: {rho[idx_x].shape}')
            print(f'rho[idx_y] shape: {rho[idx_y].shape}')
            print(f'rho[idx_z] shape: {rho[idx_z].shape}')
            print(f'ux shape: {ux.shape}')
            print(f'uy shape: {uy.shape}')
            print(f'uz shape: {uz.shape}')
            print(f'xx shape: {xx.shape}')
            print(f'yy shape: {yy.shape}')
            print(f'zz shape: {zz.shape}')

        rho_rhs = np.zeros((domain))

        momentx_rhs = np.zeros((domain))
        momenty_rhs = np.zeros((domain))
        momentz_rhs = np.zeros((domain))

        e_rhs = np.zeros((domain))
        
        # Split the operations into many if statements
        if nx > 1:
            for j in range(ny):
                for k in range(nz):
                    # get idx_xx
                    idx_xx = (slice(None), j, k)
                    idx_x = (slice(None), j, k, it)
                    
                    rho_rhs[idx_xx] += -ddx(xx, momentx[idx_x])
                    
                    momentx_rhs[idx_xx] += -(ddx(xx, momentx[idx_x]*ux[idx_x] + Pg[idx_x]))
                    momenty_rhs[idx_xx] += -(ddx(xx, rho[idx_x]*ux[idx_x]*uy[idx_x]))
                    momentz_rhs[idx_xx] += -(ddx(xx, rho[idx_x]*ux[idx_x]*uz[idx_x]))
        
                    e_rhs[idx_xx] += -ddx(xx, e[idx_x]*ux[idx_x]) - Pg[idx_x]*ddx(xx, ux[idx_x])
                        

        if ny > 1:
            for i in range(nx):
                for k in range(nz):
                    idx_yy = (i, slice(None), k)
                    idx_y = (i, slice(None), k, it)
                    
                    rho_rhs[idx_yy] += -ddx(yy, momenty[idx_y])

                    momentx_rhs[idx_yy] += -(ddx(yy, rho[idx_y]*ux[idx_y]*uy[idx_y]))
                    momenty_rhs[idx_yy] += -(ddx(yy, momenty[idx_y]*uy[idx_y] + Pg[idx_y]))
                    momentz_rhs[idx_yy] += -(ddx(yy, rho[idx_y]*uz[idx_y]*uy[idx_y]))

                    e_rhs[idx_yy] = -ddx(yy, e[idx_y]*uy[idx_y]) - Pg[idx_y]*ddx(yy, uy[idx_y])
        if nz > 1:
            for i in range(nx):
                for j in range(ny):
                    idx_zz = (i, j, slice(None))
                    idx_z = (i, j, slice(None), it)
                    
                    rho_rhs[idx_zz] += -ddx(zz, momentz[idx_z])
            
                    momentx_rhs[idx_zz] += -(ddx(zz, rho[idx_z]*ux[idx_z]*uz[idx_z]))
                    momenty_rhs[idx_zz] += -(ddx(zz, rho[idx_z]*uy[idx_z]*uz[idx_z]))
                    momentz_rhs[idx_zz] += -(ddx(zz, momentz[idx_z]*uz[idx_z] + Pg[idx_z]))
                    
                    e_rhs[idx_zz] += -ddx(zz, e[idx_z]*uz[idx_z]) - Pg[idx_z]*ddx(zz, uz[idx_z])

        if method == 'FTCS':
            """
            For solving with Forward in Time Central Scheme
            """
            rho_temp = rho[idx] + rho_rhs*dt
            
            momentx_temp = momentx[idx] + momentx_rhs*dt
            momenty_temp = momenty[idx] + momenty_rhs*dt
            momentz_temp = momentz[idx] + momentz_rhs*dt
            
            e_temp = e[idx] + e_rhs*dt

        elif method == 'LAX':
            rho_lax = np.zeros((domain))
            
            momentx_lax = np.zeros((domain))
            momenty_lax = np.zeros((domain))
            momentz_lax = np.zeros((domain))
            
            e_lax = np.zeros((domain))
            count = 0
            if nx > 1:
                count += 1
                for j in range(ny):
                    for k in range(nz):
                        idx_xx = (slice(None), j, k)
                        idx_x = (slice(None), j, k, it)

                        rho_lax[idx_xx] += (np.roll(rho[idx_x], -1) + rho[idx_x] + np.roll(rho[idx_x], 1))/ 3.
                        
                        momentx_lax[idx_xx] += (np.roll(momentx[idx_x], -1) + momentx[idx_x] + np.roll(momentx[idx_x], 1))/ 3.
                        momenty_lax[idx_xx] += (np.roll(momenty[idx_x], -1) + momenty[idx_x] + np.roll(momenty[idx_x], 1))/ 3.
                        momentz_lax[idx_xx] += (np.roll(momentz[idx_x], -1) + momentz[idx_x] + np.roll(momentz[idx_x], 1))/ 3.
                        
                        e_lax[idx_xx] += (np.roll(e[idx_x], -1) + e[idx_x] + np.roll(e[idx_x], 1))/ 3.
                        
            if ny > 1:
                count += 1
                for i in range(nx):
                    for k in range(nz):
                        idx_yy = (i, slice(None), k)
                        idx_y = (i, slice(None), k, it)

                        rho_lax[idx_yy] += (np.roll(rho[idx_y], -1) + rho[idx_y] + np.roll(rho[idx_y], 1))/ 3.
                        
                        momentx_lax[idx_yy] += (np.roll(momentx[idx_y], -1) + momentx[idx_y] + np.roll(momentx[idx_y], 1))/ 3.
                        momenty_lax[idx_yy] += (np.roll(momenty[idx_y], -1) + momenty[idx_y] + np.roll(momenty[idx_y], 1))/ 3.
                        momentz_lax[idx_yy] += (np.roll(momentz[idx_y], -1) + momentz[idx_y] + np.roll(momentz[idx_y], 1))/ 3.
                        
                        
                        e_lax[idx_yy] += (np.roll(e[idx_y], -1) + e[idx_y] + np.roll(e[idx_y], 1))/ 3.
                        
            if nz > 1:
                count += 1
                for i in range(nx):
                    for j in range(ny):
                        idx_zz = (i, j, slice(None))
                        idx_z = (i, j, slice(None), it)
                
                        rho_lax[idx_zz] += (np.roll(rho[idx_z], -1) + rho[idx_z] + np.roll(rho[idx_z], 1))/ 3.
                        
                        momentx_lax[idx_zz] += (np.roll(momentx[idx_z], -1) + momentx[idx_z] + np.roll(momentx[idx_z], 1))/ 3.
                        momenty_lax[idx_zz] += (np.roll(momenty[idx_z], -1) + momenty[idx_z] + np.roll(momenty[idx_z], 1))/ 3.
                        momentz_lax[idx_zz] += (np.roll(momentz[idx_z], -1) + momentz[idx_z] + np.roll(momentz[idx_z], 1))/ 3.
                        
                        
                        e_lax[idx_zz] += (np.roll(e[idx_z], -1) + e[idx_z] + np.roll(e[idx_z], 1))/ 3.

                
            rho_temp = rho_lax/count + rho_rhs*dt

            momentx_temp = momentx_lax/count + momentx_rhs*dt
            momenty_temp = momenty_lax/count + momenty_rhs*dt
            momentz_temp = momentz_lax/count + momentz_rhs*dt

            e_temp = e_lax/count + e_rhs*dt

        find_nan(rho_rhs, 'rho_rhs', it)
        find_nan(rho[idx], 'rho[idx]', it)
        find_nan(dt, 'dt', it)
        if np.any(np.isnan(rho_temp)):
            print(f'rho_temp nan at i: {it}')
            return Pg, rho, momentx, momenty, momentz, e, time
            
        rho[idx_next] = rho_temp
        
        momentx[idx_next] = momentx_temp
        momenty[idx_next] = momenty_temp
        momentz[idx_next] = momentz_temp
        
        e[idx_next] = e_temp


        if np.any(np.isnan(rho[idx_next])):
            print(f'rho nan value i: {it}')
            return Pg, rho, momentx, momenty, momentz, e, time
        Pg[idx_next] = (γ - 1)*e[idx_next]
        time[it+1] = time[it] + dt

    return Pg, rho, momentx, momenty, momentz, e, time
