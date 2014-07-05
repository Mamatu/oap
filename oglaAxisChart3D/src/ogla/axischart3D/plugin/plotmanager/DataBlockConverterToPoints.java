package ogla.axischart3D.plugin.plotmanager;

import ogla.data.DataBlock;
import ogla.g3d.Line3D;
import ogla.g3d.Primitive;

import ogla.g3d.Triangle3D;
import ogla.math.Vector3;
import java.awt.Color;

import java.util.ArrayList;
import java.util.List;

public class DataBlockConverterToPoints extends DataBlockConverter {

    private void newPoint(DataBlock dataBlock, int row, List<Primitive> primitives) {
        if (dataBlock.columns() < 3) {
            return;
        }
        int zindex = dataBlock.columns() - 1;
        double x = dataBlock.get(row, 0).getNumber().doubleValue();
        double y = dataBlock.get(row, 1).getNumber().doubleValue();
        double z = dataBlock.get(row, zindex).getNumber().doubleValue();
        Vector3 point = new Vector3((float) x, (float) y, (float) z);
        Line3D line3D = new Line3D(point, point);
        primitives.add(line3D);
    }

    public List<Primitive> getPrimitives(DataBlock dataBlock) {
        List<Primitive> primitives = new ArrayList<Primitive>();

        for (int fa = 0; fa < dataBlock.rows(); fa++) {
            newPoint(dataBlock, fa, primitives);
        }
        return primitives;
    }
}
