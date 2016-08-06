/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.math;

/**
 *
 * @author marcin
 */
public class Vector3 extends MathStructure {

    private Complex getRef(int index) {
        if (index == 0) {
            complexTemp.set(this.x);
            return complexTemp;
        } else if (index == 1) {
            complexTemp.set(this.y);
            return complexTemp;
        } else if (index == 2) {
            complexTemp.set(this.z);
            return complexTemp;
        }
        return null;
    }

    public Complex getComplex(int index) {
        if (index == 0) {
            return new Complex(this.x);
        } else if (index == 1) {
            return new Complex(this.y);
        } else if (index == 2) {
            return new Complex(this.z);
        }

        return null;
    }

    public int getComplexesCount() {
        return 3;
    }

    @Override
    public MathStructure createCopy() {
        return new Vector3(this);
    }

    public MathStructure copy(MathStructure mathStructure) {
        Vector3 vector3 = (Vector3) mathStructure;
        vector3.set(this);
        return vector3;
    }

    public class InvalidMatrixDimensionException extends Exception {

        public InvalidMatrixDimensionException() {
            super("Invalid matrix dimesion: should be 3x3");
        }
    }

    public Vector3() {
    }

    public Vector3(float x, float y, float z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public Vector3(float[] vector) {
        this.x = vector[0];
        this.y = vector[1];
        this.z = vector[2];
    }

    public Vector3(Vector3 vertex) {
        this.x = vertex.x;
        this.y = vertex.y;
        this.z = vertex.z;
    }

    public void set(float x, float y, float z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public void set(Complex complex, float z) {
        this.x = complex.re.floatValue();
        this.x = complex.im.floatValue();
        this.z = z;
    }

    public void set(Complex complex) {
        this.x = complex.re.floatValue();
        this.x = complex.im.floatValue();
        this.z = 0;
    }

    public void set(Vector3 vec) {
        this.x = vec.x;
        this.y = vec.y;
        this.z = vec.z;
    }

    public void add(float x, float y, float z) {
        this.x += x;
        this.y += y;
        this.z += z;
    }

    public void add(Vector3 vec) {
        this.x += vec.x;
        this.y += vec.y;
        this.z += vec.z;
    }

    public void substract(Vector3 vec) {
        this.x -= vec.x;
        this.y -= vec.y;
        this.z -= vec.z;
    }

    public void multiply(Vector3 vec) {
        this.x *= vec.x;
        this.y *= vec.y;
        this.z *= vec.z;
    }

    public void multiply(float x, float y, float z) {
        this.x *= x;
        this.y *= y;
        this.z *= z;
    }

    public void crossProduct(Vector3 vec) {
        float x1 = this.x;
        float y1 = this.y;
        float z1 = this.z;
        float x = y1 * vec.z - z1 * vec.y;
        float y = z1 * vec.x - x1 * vec.z;
        float z = x1 * vec.y - y1 * vec.x;
        this.set(x, y, z);
    }

    public void crossProduct(float x, float y, float z) {
        float ox = this.y * z - this.z * y;
        float oy = this.z * x - this.x * z;
        float oz = this.x * y - this.y * x;
        this.set(ox, oy, oz);
    }

    public void crossProduct(Vector3 vec, Vector3 outcome) {
        outcome.x = this.y * vec.z - this.z * vec.y;
        outcome.y = this.z * vec.x - this.x * vec.z;
        outcome.z = this.x * vec.y - this.y * vec.x;
    }

    public void projectionOnVector(Vector3 vector) {
        float d = this.dotProduct(vector);
        float length = vector.length();
        d = d / (length);
        this.set(vector.x * d, vector.y * d, vector.z * d);
    }

    public float dotProduct(Vector3 vec) {
        return this.x * vec.x + this.y * vec.y + this.z * vec.z;
    }

    public float dotProduct(float x, float y, float z) {
        return this.x * x + this.y * y + this.z * z;
    }

    public float get(int index) {
        if (index <= 0) {
            return x;
        } else if (index == 1) {
            return y;
        } else if (index >= 2) {
            return z;
        } else {
            return z;
        }
    }

    public Vector3 set(float v, int index) {
        if (index == 0) {
            this.x = v;
        } else if (index == 1) {
            this.y = v;
        } else if (index == 2) {
            this.z = v;
        }
        return this;
    }

    private float[][] tempRotateMatrix = new float[3][3];

    public void rotate(Matrix matrix) throws InvalidMatrixDimensionException {
        tempRotateMatrix = matrix.get(tempRotateMatrix);
        this.rotate(tempRotateMatrix, this);
    }

    public Vector3 rotate(Matrix matrix, Vector3 out) throws InvalidMatrixDimensionException {
        tempRotateMatrix = matrix.get(tempRotateMatrix);
        return this.rotate(tempRotateMatrix, out);
    }

    public Vector3 rotate(float[][] matrix, Vector3 out) throws InvalidMatrixDimensionException {
        if (matrix.length != 3) {
            throw new InvalidMatrixDimensionException();
        }

        for (int fa = 0; fa < 3; fa++) {
            if (matrix[fa].length != 3) {
                throw new InvalidMatrixDimensionException();
            }
        }
        Vector3 temp = new Vector3();
        temp.set(this);
        for (int fa = 0; fa < 3; fa++) {
            float v = 0.f;
            for (int fb = 0; fb < 3; fb++) {
                final float t = temp.get(fb) * matrix[fa][fb];
                v += t;
            }
            out.set(v, fa);
        }
        return out;
    }

    public Vector3 rotate(float[][] matrix) throws InvalidMatrixDimensionException {
        Vector3 out = new Vector3();
        return rotate(matrix, out);
    }

    private final float cos(float a) {
        return (float) Math.cos((double) a);
    }

    private final float sin(float a) {
        return (float) Math.sin((double) a);
    }

    public void rotateX(float a) {
        final float x = this.x;
        final float y = cos(a) * this.y - sin(a) * this.z;
        final float z = sin(a) * this.y + cos(a) * this.z;
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public void rotateY(float a) {
        final float x = this.x * cos(a) + sin(a) * this.z;
        final float y = this.y;
        final float z = -sin(a) * this.x + cos(a) * this.z;
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public void rotateZ(float a) {
        final float x = this.x * cos(a) - sin(a) * this.y;
        final float y = sin(a) * this.x + cos(a) * this.y;
        final float z = this.z;
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public void normalize() {

        float length = length();
        if (length != 0) {
            this.x = this.x / length;
            this.y = this.y / length;
            this.z = this.z / length;
        }
    }
    private Matrix projectionMatrix = new Matrix(3, 3);
    private Matrix projectionMatrix1 = new Matrix(3, 3);
    private Vector3 vec = null;
    private Vector3 vec1 = null;

    public Vector3 projectOnPlane(Vector3 normal, Vector3 out) {
        if (vec == null) {
            vec = new Vector3();
        }
        if (vec1 == null) {
            vec1 = new Vector3();
        }
        vec.set(this);
        float l = vec.length();
        vec.normalize();
        vec.crossProduct(normal);
        vec1.set(normal);
        vec1.crossProduct(vec);
        vec1.x = vec1.x * l;
        vec1.y = vec1.y * l;
        vec1.z = vec1.z * l;
        out.set(vec1);
        return out;
    }

    public int getCount(int[] index) {
        if (index == null) {
            return 3;
        }
        return -1;
    }

    public int getDimension() {
        return 1;
    }

    public int getSize(int index) {
        if (index == 0) {
            return 1;
        }
        return -1;
    }

    public Complex getComplex(int[] index, Complex complex) {
        if (index != null && index.length == 1) {
            if (this.getRef(index[0]) != null) {
                complex.set(this.getRef(index[0]));
            }
        }
        return complex;
    }

    public float length() {
        return (float) Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
    }

    public float length2() {
        return this.x * this.x + this.y * this.y + this.z * this.z;
    }

    public float distance(Vector3 vec1) {
        if (this == vec1) {
            return 0;
        }
        float xx = (this.x - vec1.x) * (this.x - vec1.x);
        float yy = (this.y - vec1.y) * (this.y - vec1.y);
        float zz = (this.z - vec1.z) * (this.z - vec1.z);
        return (float) Math.sqrt(xx + yy + zz);
    }

    @Override
    public boolean equals(Object object) {
        if (this == object) {
            return true;
        }
        if (!(object instanceof Vector3)) {
            return false;
        }
        Vector3 vec = (Vector3) object;
        return (this.x == vec.x && this.y == vec.y && this.z == vec.z);
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 53 * hash + Float.floatToIntBits(this.x);
        hash = 53 * hash + Float.floatToIntBits(this.y);
        hash = 53 * hash + Float.floatToIntBits(this.z);
        return hash;
    }

    @Override
    public String toString() {
        return "[" + String.valueOf(x) + "," + String.valueOf(y) + "," + String.valueOf(z) + "]";
    }
    public float x;
    public float y;
    public float z;
    private Complex complexTemp = new Complex();
}
