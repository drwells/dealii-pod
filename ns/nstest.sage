#!/usr/bin/env sage
# -*- encoding: utf-8 -*-

import sage.all as sg

def main():
    x, y = sg.var('x, y')
    a = sg.chebyshev_T(1, x)*sg.chebyshev_T(4, y)
    b = sg.chebyshev_T(1, x)*sg.chebyshev_T(4, y)
    c = sg.chebyshev_T(1, x)*sg.chebyshev_T(4, y)

    print sg.integrate(sg.integrate(a*b*c.diff(x), x, -1, 1), y, -1, 1).N(64)

if __name__ == '__main__':
    main()
