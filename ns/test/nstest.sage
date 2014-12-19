#!/usr/bin/env sage
# -*- encoding: utf-8 -*-

import sage.all as sg

def main():
    x, y, z = sg.var('x, y, z')
    orders = [[1, 4], [3, 6], [5, 8]]
    a = sg.chebyshev_T(orders[0][0], x)*sg.chebyshev_T(orders[0][1], y)
    a = [a, a, 0*x]
    b = sg.chebyshev_T(orders[1][0], x)*sg.chebyshev_T(orders[1][1], y)
    b = [b, b, 0*x]
    c = sg.chebyshev_T(orders[2][0], x)*sg.chebyshev_T(orders[2][1], y)
    c = [c, c, 0*x]

    # print sg.integrate(sg.integrate(a*b*c.diff(x), x, -1, 1), y, -1, 1).N(64)
    convection = (a[0]*(b[0]*c[0].diff(x) + b[1]*c[0].diff(y) + b[2]*c[0].diff(z))
                  + a[1]*(b[0]*c[1].diff(x) + b[1]*c[1].diff(y) + b[2]*c[1].diff(y))
                  + a[2]*(b[0]*c[2].diff(x) + b[1]*c[2].diff(y) + b[2]*c[2].diff(y)))
    print sg.integrate(sg.integrate(convection, -1, 1), -1, 1).N(64)

if __name__ == '__main__':
    main()
