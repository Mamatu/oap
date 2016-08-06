package ogla.math;

public class Quaternion extends MathStructure {

    public Vector3 xyz = new Vector3();
    public float w = 0;

    public Quaternion() {
    }

    public Quaternion(float w, Vector3 xyz) {
        this.w = w;
        this.xyz.set(xyz);
    }

    public Quaternion(float w, float x, float y, float z) {
        this.w = w;
        this.xyz.set(x, y, z);
    }

    public Quaternion(Quaternion q) {
        this.w = q.w;
        this.xyz.set(q.xyz.x, q.xyz.y, q.xyz.z);
    }

    public Matrix toMatrix() {
        Matrix matrix = new Matrix(3, 3);
        return toMatrix(matrix);
    }

    public void normalize() {
        float s = this.w * this.w + this.xyz.x * this.xyz.x
                + this.xyz.y * this.xyz.y + this.xyz.z * this.xyz.z;
        s = (float) Math.sqrt(s);
        this.w = this.w / s;
        this.xyz.x = this.xyz.x / s;
        this.xyz.y = this.xyz.y / s;
        this.xyz.z = this.xyz.z / s;
    }

    /**
     * Conversion from euler angles to quaternion representation. In Tait -
     * Bryan angles representation params are yaw, pitch, roll
     *
     * @param pop vector whose x - yaw, psi, y - omega, pitch, z - phi, roll
     */
    public void fromEulerAngles(Vector3 pop) {
        fromEulerAngles(pop.x, pop.y, pop.z);
    }

    /**
     * Conversion from euler angles to quaternion representation. In Tait -
     * Bryan angles representation params are yaw, pitch, roll
     *
     * @param psi yaw, Rz
     * @param omega pitch, Ry
     * @param phi roll, Rx
     */
    public void fromEulerAngles(float psi, float omega, float phi) {
        final float cosphi2 = (float) Math.cos((double) phi / 2.);
        final float sinphi2 = (float) Math.sin((double) phi / 2.);
        final float cospsi2 = (float) Math.cos((double) psi / 2.);
        final float sinpsi2 = (float) Math.sin((double) psi / 2.);
        final float cosomega2 = (float) Math.cos((double) omega / 2.);
        final float sinomega2 = (float) Math.sin((double) omega / 2.);
        float w = cosphi2 * cosomega2 * cospsi2 + sinphi2 * sinomega2 * sinpsi2;
        float x = sinphi2 * cosomega2 * cospsi2 - cosphi2 * sinomega2 * sinpsi2;
        float y = cosphi2 * sinomega2 * cospsi2 + sinphi2 * cosomega2 * sinpsi2;
        float z = cosphi2 * cosomega2 * sinpsi2 - sinphi2 * sinomega2 * cospsi2;

        this.w = w;
        this.xyz.x = x;
        this.xyz.y = y;
        this.xyz.z = z;
    }

    public void fromAxisAngles(float angle, Vector3 xyz) {
        fromAxisAngles(angle, xyz.x, xyz.y, xyz.z);
    }

    public void fromAxisAngles(float angle, float x, float y, float z) {
        angle = angle * 0.5f;
        Vector3 vn = new Vector3(x, y, z);
        vn.normalize();

        float sinAngle = (float) Math.sin((double) angle);

        x = (vn.x * sinAngle);
        y = (vn.y * sinAngle);
        z = (vn.z * sinAngle);
        float w = (float) Math.cos((double) angle);
        this.w = w;
        this.xyz.x = x;
        this.xyz.y = y;
        this.xyz.z = z;
    }

    public Matrix toMatrix(Matrix matrix) {
        if (matrix.columns() != 3 || matrix.rows() != 3) {
        }
        final float qx = xyz.x;
        final float qy = xyz.y;
        final float qz = xyz.z;
        final float qw = w;

        final float qx2 = xyz.x * xyz.x;
        final float qy2 = xyz.y * xyz.y;
        final float qz2 = xyz.z * xyz.z;
        final float qw2 = w * w;

        float[][] a = {
            {qw2 + qx2 - qy2 - qz2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw},
            {2 * qx * qy + 2 * qz * qw, qw2 - qx2 + qy2 - qz2, 2 * qy * qz - 2 * qx * qw},
            {2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, qw2 - qx2 - qy2 + qz2}
        };
        matrix.set(a);
        return matrix;
    }

    public void multiply(Quaternion q) {
        final float a = this.w;
        final float b = this.xyz.x;
        final float c = this.xyz.y;
        final float d = this.xyz.z;

        final float e = q.w;
        final float f = q.xyz.x;
        final float g = q.xyz.y;
        final float h = q.xyz.z;

        final float w = a * e - b * f - c * g - d * h;
        final float x = a * f + b * e + c * h - d * g;
        final float y = a * g - b * h + c * e + d * f;
        final float z = a * h + b * g - c * f + d * e;
        this.w = w;
        this.xyz.set(x, y, z);
    }

    /**
     * Conversion from quaternion to axis angle.
     *
     * @param outcome - vector to which is saved information about rotation axis
     * @return angle of rotation
     */
    public float getAxisAngles(Vector3 outcome) {
        float scale = this.xyz.length();
        outcome.x = this.xyz.x / scale;
        outcome.y = this.xyz.y / scale;
        outcome.z = this.xyz.z / scale;
        return (float) Math.acos((double) w) * 2.0f;
    }

    /**
     * Conversion from quaternion to euler angles. Angles are inserted into
     * vector [phi, omega, psi], where psi - yaw, Rz, omega - pitch, Ry, phi -
     * roll, Rx.
     *
     * @param outcome
     * @retrun outcome
     */
    public Vector3 getEulerAngles(Vector3 outcome) {
        final double q0 = w;
        final double q1 = this.xyz.x;
        final double q2 = this.xyz.y;
        final double q3 = this.xyz.z;

        final float phi = (float) Math.atan(2. * (q0 * q1 + q2 * q3) / (1. - 2. * (q1 * q1 + q2 * q2)));
        final float omega = (float) Math.asin(2. * (q0 * q2 - q3 * q1));
        final float psi = (float) Math.atan(2. * (q0 * q3 + q1 * q2) / (1. - 2. * (q2 * q2 + q3 * q3)));

        outcome.set(phi, omega, psi);
        return outcome;
    }

    @Override
    public String toString() {
        return "[" + String.valueOf(w) + ", " + String.valueOf(xyz.x) + ", " + String.valueOf(xyz.y) + ", " + String.valueOf(xyz.z) + "]";
    }

    public Complex getComplex(int index) {
        if (index > 0 && index < 3) {
            return xyz.getComplex(index);
        } else if (index == 3) {
            return new Complex(this.w);
        } else {
            return null;
        }
    }

    public int getDimension() {
        return 1;
    }

    public int getSize(int index) {
        if (index == 0) {
            return 4;
        }
        return -1;
    }

    public int getCount(int[] index) {
        if (index == null) {
            return 4;
        }
        return -1;
    }

    public Complex getComplex(int[] index, Complex complex) {
        if (index != null && index.length == 1) {
            Complex complex1 = this.getComplex(index[0]);
            if (complex1 != null) {
                complex.set(complex1);
            }
        }
        return complex;
    }

    public void set(float w, Vector3 vec) {
        this.w = w;
        this.xyz.set(vec);
    }

    public void set(Quaternion quaternion) {
        this.w = quaternion.w;
        this.xyz.set(quaternion.xyz);
    }

    @Override
    public MathStructure createCopy() {
        return new Quaternion(this.w, this.xyz);
    }

    public MathStructure copy(MathStructure mathStructure) {
        Quaternion quaternion = (Quaternion) mathStructure;
        quaternion.set(this);
        return quaternion;
    }

}
