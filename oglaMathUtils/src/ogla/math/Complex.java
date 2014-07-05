package ogla.math;

/**
 * Class represents complex number.
 *
 * @author marcin
 */
public class Complex extends MathStructure {

    public Number re = new Double(0);
    public Number im = new Double(0);

    public Complex() {
    }

    public Complex(Complex c) {
        this.re = c.re.doubleValue();
        this.im = c.im.doubleValue();
    }

    public double abs() {
        if (this.im.doubleValue() == 0) {
            return this.re.doubleValue();
        } else if (this.re.doubleValue() == 0) {
            return -this.im.doubleValue();
        } else {
            return Math.sqrt(this.re.doubleValue() * this.re.doubleValue() - this.im.doubleValue() * this.im.doubleValue());
        }
    }

    @Override
    public boolean equals(Object o) {
        if (o instanceof Complex) {
            return false;
        }
        Complex c = (Complex) o;
        if (c.im.equals(this.im) && c.re.equals(this.re)) {
            return true;
        }
        return false;
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 89 * hash + (this.re != null ? this.re.hashCode() : 0);
        hash = 89 * hash + (this.im != null ? this.im.hashCode() : 0);
        return hash;
    }

    private static int irationalCountUnit(String v) {
        int c = 0;
        for (int fa = 0; fa < v.length(); fa++) {
            if (v.charAt(fa) == 'i') {
                c++;
            }
        }
        return c;
    }

    private static Complex temp1 = new Complex();

    /**
     * Parsing of compex number. Warning! Param of this method can be imaginary
     * unit (with 'i') or real part of complex number. Param can't be complex
     * number merged by addition operator.
     *
     * @param v - imaginary unit or real part
     * @return instance of complex number
     */
    public static Complex parseComplex(String v) {
        temp1.re = 0;
        temp1.im = 0;
        Complex complex = null;
        try {
            if (v.contains("i")) {
                if (v.length() == 1) {
                    temp1.im = 1;
                    return new Complex(temp1);
                }
                String[] vs = v.split("i");
                double a = 1;
                for (String s : vs) {
                    if (s != null && s.length() > 0) {
                        a = a * Double.valueOf(s);
                    }
                }
                int ia = irationalCountUnit(v);
                if (ia % 2 == 1) {
                    temp1.im = a;
                } else {
                    temp1.re = a;
                }
            } else {
                temp1.re = Double.valueOf(v);
            }
        } catch (Exception ex) {
            return null;
        }
        return new Complex(temp1);
    }

    public Complex(Number re, Number im) {
        this.re = re;
        this.im = im;
    }

    public Complex(Number re) {
        this.re = re;
        this.im = 0;
    }

    public void set(Complex c) {
        this.re = c.re;
        this.im = c.im;
    }

    public void set(Number n) {
        this.re = n;
        this.im = 0;
    }

    private void checkRange() {
        if (Double.isInfinite(this.re.doubleValue())) {
            this.re = Double.MAX_VALUE;
        }
        if (Double.isInfinite(this.im.doubleValue())) {
            this.im = Double.MAX_VALUE;
        }
    }

    public void add(Complex c) {
        this.re = this.re.doubleValue() + c.re.doubleValue();
        this.im = this.im.doubleValue() + c.im.doubleValue();
        checkRange();
    }

    public void add(float f) {
        this.re = this.re.floatValue() + f;
    }

    public void add(Number n) {
        this.re = this.re.doubleValue() + n.doubleValue();
    }

    public void substract(Complex c) {
        this.re = this.re.doubleValue() - c.re.doubleValue();
        this.im = this.im.doubleValue() - c.im.doubleValue();
        checkRange();
    }

    public void substract(float c) {
        this.re = this.re.floatValue() - c;
        checkRange();
    }

    public void substract(Number n) {
        this.re = this.re.doubleValue() - n.doubleValue();
        checkRange();
    }

    public void multiply(float f) {
        this.re = this.re.floatValue() * f;
        checkRange();
    }

    public void multiply(Number n) {
        this.re = this.re.doubleValue() * n.doubleValue();
        checkRange();
    }

    public void multiply(Complex c) {
        double re = this.re.doubleValue() * c.re.doubleValue() - this.im.doubleValue() * c.im.doubleValue();
        double im = this.re.doubleValue() * c.im.doubleValue() + this.im.doubleValue() * c.re.doubleValue();
        this.re = re;
        this.im = im;
        checkRange();
    }

    public void divide(float n) {
        this.re = this.re.floatValue() / n;
    }

    public void divide(Number n) {
        this.re = this.re.doubleValue() / n.doubleValue();
    }

    public void divide(Complex c) {
        double u = c.re.doubleValue() * c.re.doubleValue() + c.im.doubleValue() * c.im.doubleValue();
        Complex o = new Complex();
        Complex o1 = new Complex();
        o.set(this);
        o1.set(c);
        o1.conjugate();
        o.multiply(o1);
        if (Double.isInfinite(u)) {
            o.re = 0;
            o.im = 0;
            this.re = o.re.doubleValue();
            this.im = o.im.doubleValue();
            return;
        }
        if (u == 0.d) {
            o.re = Double.MAX_VALUE;
            o.im = Double.MAX_VALUE;
            this.re = o.re.doubleValue();
            this.im = o.im.doubleValue();
            return;
        }
        o.re = o.re.doubleValue() / u;
        o.im = o.im.doubleValue() / u;
        this.re = o.re.doubleValue();
        this.im = o.im.doubleValue();
        checkRange();
    }

    public double arg() {
        double x = this.re.doubleValue();
        double y = this.im.doubleValue();

        double a = x / (Math.sqrt(x * x + y * y));
        a = Math.acos(a);
        if (y < 0) {
            a = -a;
        }
        return a;
    }

    public void conjugate() {
        this.im = -this.im.doubleValue();
    }

    @Override
    public String toString() {
        if (this.im.doubleValue() == 0) {
            return String.valueOf(this.re);
        }
        return String.valueOf(this.re) + "+" + String.valueOf(this.im) + "i";
    }

    public Complex getComplex(int[] index, Complex complex) {
        if (index.length == 1 && index[0] == 0 || index == null) {
            complex.set(this);
        }
        return complex;
    }

    public int getDimension() {
        return 1;
    }

    public int getSize(int index) {
        if (index != 0) {
            return -1;
        }
        return 1;
    }

    @Override
    public MathStructure createCopy() {
        return new Complex(this.re, this.im);
    }

    public MathStructure copy(MathStructure item) {
        Complex complex = (Complex) item;
        complex.set(this);
        return complex;
    }
}
