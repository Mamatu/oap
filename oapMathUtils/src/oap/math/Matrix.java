package ogla.math;

import java.util.List;

public class Matrix extends MathStructure {

    @Override
    public MathStructure createCopy() {
        return new Matrix(this);
    }

    @Override
    public MathStructure copy(MathStructure mathStructure) {
        Matrix matrix = (Matrix) mathStructure;
        matrix.set(this);
        return matrix;
    }

    static class Craetor implements MathStructure.Creator {

        private Brackets bracket = new Brackets() {

            public char getLeftSymbol() {
                return '[';
            }

            public char getRightSymbol() {
                return ']';
            }
        };

        public Brackets getBoundary() {
            return bracket;
        }

        protected Matrix matrix = null;
        protected Complex complex = null;

        private Complex createComplex(List<MathStructure> mathStructures) {
            if (mathStructures.size() == 1) {
                if (mathStructures.get(0).getSize(0) == 1) {
                    Complex complex = new Complex();
                    int[] index1 = {0};
                    complex = mathStructures.get(0).getComplex(index1, complex);
                    return complex;
                }
            }
            return null;
        }

        private Matrix createMatrix(final List<MathStructure> mathStructures) throws SyntaxErrorException {
            int dimension = -1;
            int rows = 0;
            int columns = 0;
            int columnsPrev = 0;
            boolean first = true;
            for (MathStructure mathStructure : mathStructures) {
                if (mathStructure.getDimension() == dimension || first == true) {

                    dimension = mathStructure.getDimension();
                    if (dimension == 1) {
                        columns = mathStructure.getSize(0);
                    } else if (dimension == 2) {
                        rows = mathStructure.getSize(0);
                        columns = mathStructure.getSize(1);

                    }
                    if (!first) {
                        if (columnsPrev != columns) {
                            throw new SyntaxErrorException("Invalid columns");
                        }
                    }
                    first = false;
                    columnsPrev = columns;
                }
            }
            final int dimesnion2 = mathStructures.size() == 1 ? 1 : 2;
            final int columns1 = columns;
            final int rows1 = rows;
            MathStructure mathStructure1 = new MathStructure() {
                public Complex getComplex(int[] index, Complex out) {
                    if (index != null) {
                        if (dimesnion2 == 2) {
                            final int columnsIndex = index[1];
                            int rowIndex = index[0] % (rows1);
                            int msIndex = index[0] / rows1;
                            int[] index1 = {rowIndex, columnsIndex};
                            out = mathStructures.get(msIndex).getComplex(index1, out);
                        } else if (dimesnion2 == 1) {
                            int[] index1 = {index[0]};
                            out = mathStructures.get(0).getComplex(index1, out);
                        }
                        return out;
                    }
                    return out;
                }

                public int getDimension() {
                    return dimesnion2;
                }

                public int getSize(int index) {
                    if (index == 0) {
                        return mathStructures.size();
                    } else if (index == 1) {
                        if (mathStructures.get(0).getDimension() == 2) {
                            return mathStructures.get(0).getSize(1);
                        } else {
                            return mathStructures.get(0).getSize(0);
                        }
                    }
                    return -1;
                }

                @Override
                public MathStructure createCopy() {
                    throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
                }

                public MathStructure copy(MathStructure mathStructure) {
                    throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
                }
            };
            return this.createMatrix(mathStructure1);
        }

        private Matrix createMatrix(MathStructure mathStructure) {
            int rows = mathStructure.getSize(0);
            int columns = -1;
            for (int fa = 0; fa < rows; fa++) {
                int columns1 = mathStructure.getSize(1);
                if (columns == -1) {
                    columns = columns1;
                }
                if (columns != columns1) {
                    return null;
                }
            }
            if (columns <= 0) {
                return null;
            }
            Matrix matrix = new Matrix(rows, columns);
            matrix.set(mathStructure);
            return matrix;
        }

        public boolean setParams(List<MathStructure> mathStructure) throws SyntaxErrorException {
            this.matrix = null;
            Matrix matrix = this.createMatrix(mathStructure);
            if (matrix == null) {
                return false;
            }
            this.matrix = matrix;
            return true;
        }

        public MathStructure create() {
            if (matrix != null) {
                return matrix;
            }
            if (complex != null) {
                return complex;
            }
            return null;
        }

    }

    public static Matrix createUnitMatrix(int rows, int columns) {
        Matrix m = new Matrix(rows, columns);
        for (int fa = 0; fa < rows; fa++) {
            for (int fb = 0; fb < columns; fb++) {
                if (fa == fb) {
                    m.array[fa * columns + fb] = new Complex(1);
                } else {
                    m.array[fa * columns + fb] = new Complex(0);
                }
            }
        }
        return m;
    }

    private static float cos(float angle) {
        return (float) Math.cos((double) angle);
    }

    private static float sin(float angle) {
        return (float) Math.sin((double) angle);
    }

    public static Matrix createRotationMatrixX(float angle) {
        Matrix m = new Matrix(3, 3);
        float cosa = cos(angle);
        float sina = sin(angle);
        float[][] array = {{1.f, 0.f, 0.f}, {0.f, cosa, -sina}, {0.f, sina, cosa}};
        m.set(array);
        return m;
    }

    public static Matrix createRotationMatrixY(float angle) {
        Matrix m = new Matrix(3, 3);
        float cosa = cos(angle);
        float sina = sin(angle);
        float[][] array = {{cosa, 0, sina}, {0, 1, 0}, {-sina, 0, cosa}};
        m.set(array);
        return m;
    }

    public static Matrix createRotationMatrixZ(float angle) {
        Matrix m = new Matrix(3, 3);
        float cosa = cos(angle);
        float sina = sin(angle);
        float[][] array = {{cosa, -sina, 0}, {sina, cosa, 0}, {0, 0, 1}};
        m.set(array);
        return m;
    }

    Complex[] array = null;
    Complex[] array1 = null;

    private void createTempArrayCopy() {
        if (this.array1 == null) {
            this.array1 = new Complex[rows * columns];
        }
        System.arraycopy(this.array, 0, this.array1, 0, this.rows * this.columns);
    }

    private int rows = 0, columns = 0;

    private Complex getRef(int x, int y) {
        return array[x * this.columns + y];
    }

    private Complex get1(int x, int y) {
        return array1[x * this.columns + y];
    }

    public float[][] get(float[][] array) {
        if (array.length == this.rows) {
            for (int fa = 0; fa < this.rows; fa++) {
                for (int fb = 0; fb < this.columns; fb++) {
                    array[fa][fb] = this.array[fa * this.columns + fb].re.floatValue();
                }
            }
        }
        return array;
    }

    public int rows() {
        return rows;
    }

    public int columns() {
        return columns;
    }

    public Matrix(int rows, int columns) {
        array = new Complex[rows * columns];
        for (int fa = 0; fa < rows * columns; fa++) {
            this.array[fa] = new Complex(0);
        }
        this.rows = rows;
        this.columns = columns;
    }

    public Matrix(Matrix orig) {
        array = new Complex[orig.rows * orig.columns];
        for (int fa = 0; fa < orig.rows * orig.columns; fa++) {
            this.array[fa] = new Complex(orig.array[fa]);
        }
        this.rows = orig.rows;
        this.columns = orig.columns;
    }

    public void set(Matrix orig) {
        if (orig.columns != this.columns || orig.rows != this.rows) {
            array = new Complex[orig.rows * orig.columns];
            for (int fa = 0; fa < orig.rows * orig.columns; fa++) {
                this.array[fa] = new Complex(orig.array[fa]);
            }
            this.rows = orig.rows;
            this.columns = orig.columns;
        } else {
            this.set(orig.array);
        }
    }

    public void set(float[][] i) {
        for (int fa = 0; fa < rows; fa++) {
            for (int fb = 0; fb < columns; fb++) {
                this.getRef(fa, fb).set(i[fa][fb]);
            }
        }
    }

    public void set(Number[][] i) {
        for (int fa = 0; fa < rows; fa++) {
            for (int fb = 0; fb < columns; fb++) {
                this.getRef(fa, fb).set(i[fa][fb].doubleValue());
            }
        }
    }

    public void set(Complex[][] i) {
        for (int fa = 0; fa < rows; fa++) {
            for (int fb = 0; fb < columns; fb++) {
                this.getRef(fa, fb).set(i[fa][fb]);
            }
        }
    }

    public void set(MathStructure mathStructure) {
        if (mathStructure.getDimension() > 2 || mathStructure.getDimension() <= 0) {
            return;
        }
        if (mathStructure.getDimension() == 2) {
            for (int fa = 0; fa < mathStructure.getSize(0); fa++) {
                for (int fb = 0; fb < mathStructure.getSize(1); fb++) {
                    int[] index1 = {fa, fb};
                    Complex complex = this.getRef(fa, fb);
                    complex = mathStructure.getComplex(index1, complex);
                }
            }
        } else if (mathStructure.getDimension() == 1 && this.rows == 1) {
            for (int fa = 0; fa < mathStructure.getSize(1); fa++) {
                Complex complex = this.getRef(0, fa);
                int[] index1 = {fa};
                complex = mathStructure.getComplex(index1, complex);
            }
        }
    }

    public void set(float[] i) {
        for (int fa = 0; fa < rows; fa++) {
            for (int fb = 0; fb < columns; fb++) {
                this.getRef(fa, fb).set(i[fa * this.columns + fb]);
            }
        }
    }

    public void set(Number[] i) {
        for (int fa = 0; fa < rows; fa++) {
            for (int fb = 0; fb < columns; fb++) {
                this.getRef(fa, fb).set(i[fa * this.columns + fb].doubleValue());
            }
        }
    }

    public void set(Complex[] i) {
        for (int fa = 0; fa < rows; fa++) {
            for (int fb = 0; fb < columns; fb++) {
                this.getRef(fa, fb).set(i[fa * this.columns + fb]);
            }
        }
    }

    public void set(Number v, int r, int c) {
        this.getRef(r, c).set(v);
    }

    public void set(float v, int r, int c) {
        this.getRef(r, c).set(v);
    }

    public void set(Complex v, int r, int c) {
        this.getRef(r, c).set(v);
    }

    public float getFloat(int r, int c) {
        return this.getRef(r, c).re.floatValue();
    }

    public float getNumber(int r, int c) {
        return this.getRef(r, c).re.floatValue();
    }

    public Complex getComplex(int r, int c) {
        Complex complex = new Complex(this.getRef(r, c));
        return complex;
    }

    public void add(Matrix matrix) {
        for (int fa = 0; fa < rows; fa++) {
            for (int fb = 0; fb < columns; fb++) {
                this.getRef(fa, fb).add(matrix.getRef(fa, fb));
            }
        }
    }

    public void substract(Matrix matrix) {
        for (int fa = 0; fa < rows; fa++) {
            for (int fb = 0; fb < columns; fb++) {
                this.getRef(fa, fb).substract(matrix.getRef(fa, fb));
            }
        }
    }

    public void transpose() {
        this.createTempArrayCopy();
        for (int fa = 0; fa < rows; fa++) {
            for (int fb = 0; fb < columns; fb++) {
                this.set(this.array1[fb * columns + fa], fb, fa);
            }
        }
    }

    public void multiply(Complex complex) {
        for (int fa = 0; fa < rows; fa++) {
            for (int fb = 0; fb < columns; fb++) {
                this.getRef(fa, fb).multiply(complex);
            }
        }
    }

    public void multiply(Matrix matrix) {
        this.createTempArrayCopy();
        for (int fa = 0; fa < this.rows; fa++) {
            for (int fb = 0; fb < matrix.columns; fb++) {
                for (int na = 0; na < matrix.rows; na++) {
                    this.getRef(fa, fb).set(this.get1(fa, na));
                    this.getRef(fa, fb).multiply(matrix.get1(na, fb));
                }
            }
        }
    }

    public Vector3 multiplyQuadricMatrix(Vector3 vector, Vector3 outcome) {
        if (this.rows != 3 || this.columns != 3) {
            return null;
        }

        Complex c00 = new Complex();
        Complex c01 = new Complex();
        Complex c02 = new Complex();

        Complex c10 = new Complex();
        Complex c11 = new Complex();
        Complex c12 = new Complex();

        Complex c20 = new Complex();
        Complex c21 = new Complex();
        Complex c22 = new Complex();

        c00.set(vector.x);
        c00.multiply(this.getRef(0, 0));
        c01.set(vector.y);
        c01.multiply(this.getRef(0, 1));
        c02.set(vector.z);
        c02.multiply(this.getRef(0, 2));

        c10.set(vector.x);
        c10.multiply(this.getRef(1, 0));
        c11.set(vector.y);
        c11.multiply(this.getRef(1, 1));
        c12.set(vector.z);
        c12.multiply(this.getRef(1, 2));

        c20.set(vector.x);
        c20.multiply(this.getRef(2, 0));
        c21.set(vector.y);
        c21.multiply(this.getRef(2, 1));
        c22.set(vector.z);
        c22.multiply(this.getRef(2, 2));

        outcome.x = (c00).re.floatValue();
        outcome.x += (c01).re.floatValue();
        outcome.x += (c02).re.floatValue();

        outcome.y = (c10).re.floatValue();
        outcome.y += (c11).re.floatValue();
        outcome.y += (c12).re.floatValue();

        outcome.z = (c20).re.floatValue();
        outcome.z += (c21).re.floatValue();
        outcome.z += (c22).re.floatValue();

        return outcome;
    }

    private Matrix cut(Matrix matrix, int i, int j, Matrix d) {

        for (int fa = 0; fa < d.rows; fa++) {
            for (int fb = 0; fb < d.columns; fb++) {
                int na, nb;
                if (fa < i) {
                    na = fa;
                } else {
                    na = fa + 1;
                }
                if (fb < j) {
                    nb = fb;
                } else {
                    nb = fb + 1;
                }
                d.getRef(fa, fb).set(matrix.getRef(na, nb));
            }
        }
        return d;
    }

    public final void fillMatrix(float v) {
        for (int fa = 0; fa < this.rows; fa++) {
            for (int fb = 0; fb < this.columns; fb++) {
                this.getRef(fa, fb).set(v);
            }
        }
    }

    public final void prepareIdentityMatrix() {
        for (int fa = 0; fa < this.rows; fa++) {
            for (int fb = 0; fb < this.columns; fb++) {
                this.getRef(fa, fb).set(0);
                if (fa == fb) {
                    this.getRef(fa, fb).set(1);
                }
            }
        }
    }

    public final void clear() {
        fillMatrix(0);
    }

    private Matrix cut(Matrix matrix, int i, int j) {
        Matrix d = new Matrix(matrix.rows - 1, matrix.columns - 1);
        return cut(matrix, i, j, d);
    }

    private float det(Matrix matrix) {
        if (matrix.array.length == 1) {
            float d = 0.f;
            for (int fa = 0; fa < matrix.columns; fa++) {
                d += matrix.getRef(0, fa).re.floatValue();
            }
            return d;
        }

        if (matrix.columns == 1) {
            float d = 0.f;
            for (int fa = 0; fa < matrix.array.length; fa++) {
                d += matrix.getRef(fa, 0).re.floatValue();
            }
            return d;
        }
        float d = 0.f;
        int i = 0;

        for (int fa = 0; fa < matrix.columns; fa++) {
            Matrix n = cut(matrix, i, fa);
            float p = -1;
            if ((i + fa) % 2 == 0) {
                p = 1;
            }
            if (matrix.getRef(i, fa).re.floatValue() != 0) {
                d += matrix.getRef(i, fa).re.floatValue() * p * det(n);
            }
        }
        return d;
    }

    public float det() {
        return this.det(this);
    }

    public void adjugate() {
        Matrix out = new Matrix(this.rows, this.columns);
        for (int fa = 0; fa < out.rows(); fa++) {
            for (int fb = 0; fb < out.columns(); fb++) {
                float p = -1;
                if ((fa + fb) % 2 == 0) {
                    p = 1;
                }
                out.getRef(fa, fb).set(cut(this, fa, fb).det() * p);
            }
        }

        out.transpose();
        this.set(out);
    }

    public void inverse() {
        Matrix out = new Matrix(this.rows, this.columns);
        out.set(this);
        float d = out.det();

        out.adjugate();

        for (int fa = 0; fa < out.rows(); fa++) {
            for (int fb = 0; fb < out.columns(); fb++) {
                out.getRef(fa, fb).set(out.getRef(fa, fb).re.floatValue() / d);
            }
        }
        this.set(out);
    }

    @Override
    public String toString() {
        String out = "{";
        for (int fa = 0; fa < this.rows; fa++) {
            out += "{";
            for (int fb = 0; fb < this.columns; fb++) {
                out += String.valueOf(this.getRef(fa, fb));
                if (fb != this.columns - 1) {
                    out += ",";
                }
            }
            out += "}";
        }
        out += "}\n";
        return out;
    }

    public Complex getComplex(int index) {
        if (index > this.rows * this.columns || index < 0) {
            return null;
        }
        int row = index / this.columns;
        int column = index % this.columns;
        return new Complex(this.getRef(row, column));

    }

    public int getCount(int[] index) {
        if (index == null) {
            return this.rows;
        }
        if (index.length != 1) {
            return -1;
        }
        return this.columns;
    }

    public int getDimension() {
        return 2;
    }

    public int getSize(int index) {
        if (index == 0) {
            return this.rows;
        } else if (index == 1) {
            return this.columns;
        }
        return -1;
    }

    public Complex getComplex(int[] index, Complex out) {
        if (index.length == 2) {
            out.set(this.getRef(index[0], index[1]));
        } else if (index.length == 1) {
            out.set(this.getRef(0, index[0]));
        }
        return out;
    }
}
