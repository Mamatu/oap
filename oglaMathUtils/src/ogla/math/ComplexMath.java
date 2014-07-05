package ogla.math;
/**
 * ComplexMath delivers methods which are equivalent of default Java Math class.
 * Methods are extended to support complex number.
 * @author marcin
 */
public class ComplexMath {

    @FunctionDescription(description = "Calculate absolute value of complex number.",
    outcome = "Complex number whose Re is absolute value of param, Im is zero",
    params = "c - complex number")
    public static Complex abs(Complex c) {
        Complex o = new Complex();
        if (c.im.doubleValue() == 0) {
            if (c.re.doubleValue() < 0) {
                o.re = -c.re.doubleValue();
            } else {
                o.re = c.re.doubleValue();
            }
            return o;
        }
        o.re = Math.sqrt(c.re.doubleValue() * c.re.doubleValue() + c.im.doubleValue() * c.im.doubleValue());
        return o;
    }

    @FunctionDescription(description = "Calculate natural logarithm of complex number.",
    params = {"c - complex number"})
    public static Complex log(Complex c) {
        Complex o = new Complex();
        if (c.im.doubleValue() == 0) {
            o.re = Math.log(c.re.doubleValue());
            return o;
        }
        o.re = Math.log(abs(c).re.doubleValue());
        o.im = c.arg();
        return o;
    }

    @FunctionDescription(description = "Calculate logarithm with base 10 of complex number.")
    public static Complex log10(Complex c) {
        Complex o = new Complex();
        if (c.im.doubleValue() == 0) {
            o.re = Math.log10(c.re.doubleValue());
            return o;
        }
        o.re = Math.log10(abs(c).re.doubleValue());
        o.im = c.arg();
        return o;
    }

    @FunctionDescription(description = "Calculate natural logarith of c+1.")
    public static Complex log1p(Complex c) {
        Complex o = new Complex();
        if (c.im.doubleValue() == 0) {
            o.re = Math.log1p(c.re.doubleValue());
            return o;
        }
        o.re = Math.log1p(abs(c).re.doubleValue());
        o.im = c.arg();
        return o;
    }

    @FunctionDescription(description = "Exponental function.")
    public static Complex exp(Complex c) {
        Complex o = new Complex();
        final double re = Math.exp(c.re.doubleValue());
        if (c.im.doubleValue() == 0) {
            o.re = re;
            return o;
        }
        o.re = re * Math.cos(c.im.doubleValue());
        o.im = re * Math.sin(c.im.doubleValue());
        return o;
    }

    @FunctionDescription(description = "Exponental function. Calculate exp(1-c).")
    public static Complex expm1(Complex c) {
        Complex o = new Complex();
        final double re = Math.expm1(c.re.doubleValue());
        if (c.im.doubleValue() == 0) {
            o.re = re;
            return o;
        }
        o.re = re * Math.cos(c.im.doubleValue());
        o.im = re * Math.sin(c.im.doubleValue());
        return o;
    }

    @FunctionDescription(description = "Calculate a to power of b",
    params = {"a - base", "b - exponent"})
    public static Complex pow(Complex a, Complex b) {
        if (a.im.doubleValue() == 0 && b.im.doubleValue() == 0) {
            Complex c = new Complex();
            c.re = Math.pow(a.re.doubleValue(), b.re.doubleValue());
            return c;
        }
        Complex c = log(a);
        c.multiply(b);
        Complex out = exp(c);
        return out;
    }

    @FunctionDescription(description = "Sinus")
    public static Complex sin(Complex c) {
        Complex o = new Complex();
        if (c.im.doubleValue() == 0) {
            o.re = Math.sin(c.re.doubleValue());
            return o;
        }
        o.re = Math.sin(c.re.doubleValue()) * Math.cosh(c.im.doubleValue());
        o.im = Math.cos(c.re.doubleValue()) * Math.sinh(c.im.doubleValue());
        return o;
    }

    @FunctionDescription(description = "Cosinus")
    public static Complex cos(Complex c) {
        Complex o = new Complex();
        if (c.im.doubleValue() == 0) {
            o.re = Math.cos(c.re.doubleValue());
            return o;
        }
        o.re = Math.cos(c.re.doubleValue()) * Math.cosh(c.im.doubleValue());
        o.im = -Math.sin(c.re.doubleValue()) * Math.sinh(c.im.doubleValue());
        return o;
    }

    @FunctionDescription(description = "Tangent")
    public static Complex tan(Complex c) {
        Complex o = new Complex();
        if (c.im.doubleValue() == 0) {
            o.re = Math.tan(c.re.doubleValue());
            return o;
        }
        Complex rs = ComplexMath.sin(c);
        Complex rc = ComplexMath.cos(c);
        rs.divide(rc);
        return rs;
    }

    @FunctionDescription(description = "Cotangent")
    public static Complex ctan(Complex c) {
        Complex o = new Complex();
        if (c.im.doubleValue() == 0) {
            o.re = Math.tan(c.re.doubleValue());
            o.re = 1.d / o.re.doubleValue();
            return o;
        }
        Complex rs = ComplexMath.sin(c);
        Complex rc = ComplexMath.cos(c);
        rc.divide(rs);
        return rc;
    }

    @FunctionDescription(description = "Hiperbolic sinus")
    public static Complex sinh(Complex c) {
        Complex o = new Complex();
        double x = c.re.doubleValue();
        double y = c.im.doubleValue();
        if (y == 0) {
            o.re = Math.sinh(x);
            return o;
        }
        if (x == 0) {
            o.im = Math.sin(y);
            return o;
        }
        o.re = Math.sinh(x) * Math.cos(y);
        o.im = Math.cosh(x) * Math.sin(y);
        return o;
    }

    @FunctionDescription(description = "Hiperbolic cosinus")
    public static Complex cosh(Complex c) {
        Complex o = new Complex();
        double x = c.re.doubleValue();
        double y = c.im.doubleValue();
        if (y == 0) {
            o.re = Math.cosh(x);
            return o;
        }
        if (x == 0) {
            o.re = Math.cos(y);
            return o;
        }
        o.re = Math.cosh(x) * Math.cos(y);
        o.im = Math.sinh(x) * Math.sin(y);
        return o;
    }

    @FunctionDescription(description = "Hiperbolic tangent")
    public static Complex tanh(Complex c) {
        Complex o = new Complex();
        if (c.im.doubleValue() == 0) {
            o.re = Math.tanh(c.re.doubleValue());
            return o;
        }
        Complex rs = ComplexMath.sinh(c);
        Complex rc = ComplexMath.cosh(c);
        rs.divide(rc);
        return rs;
    }

    @FunctionDescription(description = "Hiperbolic cotangent")
    public static Complex ctanh(Complex c) {
        Complex o = new Complex();
        if (c.im.doubleValue() == 0) {
            o.re = Math.tanh(c.re.doubleValue());
            o.re = 1.d / o.re.doubleValue();
            return o;
        }
        Complex rs = ComplexMath.sinh(c);
        Complex rc = ComplexMath.cosh(c);
        rc.divide(rs);
        return rc;
    }

    private static double sgn(double n) {
        if (n < 0) {
            return -1;
        } else if (n == 0) {
            return 0;
        } else if (n > 0) {
            return 1;
        }
        return 1;
    }

    @FunctionDescription(description = "Square root of complex number c.",
    outcome = "square root of complex, if you want to get second square you should multiply outcome by -1.")
    public static Complex sqrt(Complex c) {
        if (c.im.doubleValue() != 0) {
            double p = Math.sqrt(c.re.doubleValue() * c.re.doubleValue() + c.im.doubleValue() * c.im.doubleValue());
            double y = Math.sqrt((c.re.doubleValue() + p) / 2.);
            double o = ComplexMath.sgn(c.im.doubleValue()) * Math.sqrt((-c.re.doubleValue() + p) / 2.);
            return new Complex(y, o);

        }
        Complex o = new Complex();
        o.re = Math.sqrt(c.re.doubleValue());
        return o;
    }

    @FunctionDescription(description = "Arc sinus")
    public static Complex asin(Complex c) {
        Complex iz = new Complex(0, 1);
        iz.multiply(c);
        Complex c1 = new Complex(1, 0);
        Complex z2 = new Complex(c);
        z2.multiply(c);
        c1.substract(z2);
        Complex s1z2 = new Complex(sqrt(c1));
        Complex sum = new Complex(iz);
        sum.add(s1z2);
        Complex o = new Complex(ComplexMath.log(sum));
        o.multiply(new Complex(0, -1));
        return o;
    }

    @FunctionDescription(description = "Arc cosinus")
    public static Complex acos(Complex c) {
        Complex c1 = new Complex(1, 0);
        Complex z2 = new Complex(c);
        z2.multiply(c);
        z2.substract(c1);
        Complex sz21 = new Complex(sqrt(z2));
        Complex sum = new Complex(c);
        sum.add(sz21);
        Complex o = new Complex(ComplexMath.log(sum));
        o.multiply(new Complex(0, -1));
        return o;
    }

    @FunctionDescription(description = "Arc tangent")
    public static Complex atan(Complex c) {
        Complex o = new Complex(0, 0.5f);
        Complex ic = new Complex(0, 1);
        ic.multiply(c);
        Complex a1 = new Complex(1, 0);
        Complex a2 = new Complex(1, 0);
        a1.substract(ic);
        Complex l1miz = ComplexMath.log(a1);
        a2.add(ic);
        Complex l1piz = ComplexMath.log(a2);
        Complex sub = new Complex(l1miz);
        sub.substract(l1piz);
        o.multiply(sub);
        return o;
    }

    @FunctionDescription(description = "Arc cotangent")
    public static Complex actan(Complex c) {
        Complex o = new Complex(0, 0.5f);
        Complex ic = new Complex(0, 1);
        ic.divide(c);
        Complex a1 = new Complex(1, 0);
        Complex a2 = new Complex(1, 0);
        a1.substract(ic);
        Complex l1miz = ComplexMath.log(a1);
        a2.add(ic);
        Complex l1piz = ComplexMath.log(a2);
        Complex sub = new Complex(l1miz);
        sub.substract(l1piz);
        o.multiply(sub);
        return o;
    }

    @FunctionDescription(description = "Get imaginary unit of complex number",
    outcome = "Complex number whose real part is imaginary unit of param. Imaginary part is zero.")
    public static Complex im(Complex c) {
        return new Complex(c.im, 0);
    }

    @FunctionDescription(description = "Get real part of complex number",
    outcome = "Complex number whose real part is real part of param. Imaginary part is zero.")
    public static Complex re(Complex c) {
        return new Complex(c.re, 0);
    }
}
