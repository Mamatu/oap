package ogla.axischart3D.plugin.plotmanager;

import ogla.math.Vector3;
import java.util.List;

public class DataBlockConverterTools {

    public static enum Axis {

        X, Y
    };

    public static enum AxisPart {

        NEGATIVE, POSITIVE
    };

    public static void setMin(Vector3 comp, Float[] min) {
        if (min[0] == null || comp.x < min[0]) {
            min[0] = comp.x;
        }
        if (min[1] == null || comp.y < min[1]) {
            min[1] = comp.y;
        }
        if (min[2] == null || comp.z < min[2]) {
            min[2] = comp.z;
        }
    }

      public static void setMax(Vector3 comp, Float[] max) {
        if (max[0] == null || comp.x > max[0]) {
            max[0] = comp.x;
        }
        if (max[1] == null || comp.y > max[1]) {
            max[1] = comp.y;
        }
        if (max[2] == null || comp.z > max[2]) {
            max[2] = comp.z;
        }
    }

    public static void fillVectors(List<Vector3> vectors, Vector3 ref, List<Vector3> out, Axis axis, AxisPart axisPart) {
        Float min = null;
        for (Vector3 vector : vectors) {
            if (vector != ref) {
                float d1 = (float)Math.sqrt((vector.x - ref.x)*(vector.x - ref.x)+(vector.y - ref.y)*(vector.y - ref.y));
                if ((min == null || d1 <= min)) {
                    if (min != null && d1 < min) {
                        out.clear();
                    }
                    min = d1;
                    out.add(vector);
                }
            }
        }
    }

    public static Vector3 getNearest(List<Vector3> vectors, Vector3 refCorner, Axis axis, AxisPart axisPart) {
        Float min = null;
        Vector3 nearest = null;
        for (Vector3 vector : vectors) {
            float d = 0;
            if (axis == Axis.X) {
                d = vector.x - refCorner.x;
            } else if (axis == Axis.Y) {
                d = vector.y - refCorner.y;
            }
            if ((axisPart == AxisPart.POSITIVE && d >= 0) || (axisPart == AxisPart.NEGATIVE && d <= 0)) {
                if (d < 0) {
                    d = -d;
                }
                if (min == null || d < min) {
                    if (nearest == null) {
                        nearest = new Vector3();
                    }
                    nearest.set(vector);
                    min = d;

                }
            }
        }
        return nearest;
    }
}
