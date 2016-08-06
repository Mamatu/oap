package ogla.axischart3D.plugin.plotmanager;

import ogla.g3d.Color;
import ogla.math.Vector3;

public abstract class Gradient {

    public enum Axis {

        X, Y, Z, ALL
    }

    public Color getColor(Vector3 vector, Vector3 min, Vector3 max, Axis axis) {
        if (axis == null) {
            return getColor(vector, min, max);
        }
        if (axis == Axis.X) {
            return getColor(vector.x, min.x, max.x, axis);
        } else if (axis == Axis.Y) {
            return getColor(vector.y, min.y, max.y, axis);
        } else if (axis == Axis.Z) {
            return getColor(vector.z, min.z, max.z, axis);
        } else {
            return getColor(vector, min, max);
        }
    }

    protected abstract Color getColor(float v, float min, float max, Axis axis);

    protected abstract Color getColor(Vector3 vector, Vector3 min, Vector3 max);
}
