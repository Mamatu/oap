package ogla.axischart3D.plugin.plotmanager;

import ogla.data.DataBlock;
import ogla.g3d.Color;
import ogla.g3d.Primitive;
import ogla.math.Vector3;
import java.util.List;

public abstract class DataBlockConverter {

    public void repaintPrimitives(Gradient gradient, Gradient.Axis axis, List<Primitive> primitives) {
        Float[] min = {null, null, null};
        Float[] max = {null, null, null};
        for (Primitive primitive : primitives) {
            for (Vector3 vec : primitive.getVertices()) {
                DataBlockConverterTools.setMax(vec, max);
                DataBlockConverterTools.setMin(vec, min);
            }
        }
        Vector3 vmin = new Vector3(min[0], min[1], min[2]);
        Vector3 vmax = new Vector3(max[0], max[1], max[2]);
        for (Primitive primitive : primitives) {
            Vector3[] vecs = primitive.getVertices();
            for (int fa = 0; fa < vecs.length; fa++) {
                Vector3 vector = vecs[fa];
                Color color = gradient.getColor(vector, vmin, vmax, axis);
                primitive.getColors()[fa].set(color);
            }
        }
    }

    protected abstract List<Primitive> getPrimitives(DataBlock dataBlock);

    public List<Primitive> get(DataBlock dataBlock, Gradient gradient, Gradient.Axis axis) {
        List<Primitive> primitives = getPrimitives(dataBlock);
        if (gradient != null) {
            repaintPrimitives(gradient, axis, primitives);
        }
        return primitives;
    }

    public List<Primitive> get(DataBlock dataBlock) {
        return get(dataBlock, null, null);
    }

    public static Vector3 setMax(Vector3 max, Vector3 comp) {
        if (max == null) {
            max = new Vector3(comp.x, comp.y, comp.z);
        }

        if (max.x < comp.x) {
            max.x = comp.x;
        }

        if (max.y < comp.y) {
            max.y = comp.y;
        }

        if (max.z < comp.z) {
            max.z = comp.z;
        }
        return max;
    }

    public static Vector3 setMin(Vector3 min, Vector3 comp) {
        if (min == null) {
            min = new Vector3(comp.x, comp.y, comp.z);
        }

        if (min.x > comp.x) {
            min.x = comp.x;
        }

        if (min.y > comp.y) {
            min.y = comp.y;
        }

        if (min.z > comp.z) {
            min.z = comp.z;
        }
        return min;
    }
}
