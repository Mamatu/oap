package ogla.axischart3D.plugin.plotmanager;

import ogla.data.DataBlock;
import ogla.g3d.Color;

import ogla.g3d.Line3D;
import ogla.g3d.Primitive;

import ogla.math.Vector3;
import java.util.ArrayList;
import java.util.List;

public class DataBlockConverterToGrid extends DataBlockConverter {

    private Line3D isSame(Vector3 v1, Vector3 v2, List<Primitive> primitives) {
        if (v1 == null || v2 == null) {
            return null;
        }
        for (Primitive primitive : primitives) {
            Line3D line = (Line3D) primitive;
            if (line.compare(v1, v2)) {
                return null;
            }
        }
        return new Line3D(v1, v2);
    }

    private void addTo(Line3D line, List<Primitive> primitives) {
        if (line != null) {
            primitives.add(line);
        }
    }

    public void getLines(List<Primitive> primitives, Vector3 refCorner, Vector3 corner1, Vector3 corner2) {
        Line3D line1 = isSame(refCorner, corner1, primitives);
        Line3D line2 = isSame(corner2, refCorner, primitives);
        addTo(line1, primitives);
        addTo(line2, primitives);
    }

    private Vector3 getVector(DataBlock dataBlock, int row) {
        float x = dataBlock.get(row, 0).getNumber().floatValue();
        float y = dataBlock.get(row, 1).getNumber().floatValue();
        float z = dataBlock.get(row, dataBlock.columns() - 1).getNumber().floatValue();
        return new Vector3(x, y, z);
    }

    @Override
    protected List<Primitive> getPrimitives(DataBlock dataBlock) {
        List<Primitive> primitives = new ArrayList<Primitive>();
        List<Vector3> vectors = new ArrayList<Vector3>();
        Float[] min = {null, null, null};
        Float[] max = {null, null, null};
        for (int fa = 0; fa < dataBlock.rows(); fa++) {
            Vector3 vec = getVector(dataBlock, fa);
            vectors.add(vec);
            DataBlockConverterTools.setMax(vec, max);
            DataBlockConverterTools.setMin(vec, min);
        }
        Vector3 vmin = new Vector3(min[0], min[1], min[2]);
        Vector3 vmax = new Vector3(max[0], max[1], max[2]);
        for (int fa = 0; fa < vectors.size(); fa++) {
            Vector3 vector = vectors.get(fa);
            List<Vector3> out = new ArrayList<Vector3>();
            DataBlockConverterTools.fillVectors(vectors, vector, out, DataBlockConverterTools.Axis.X, DataBlockConverterTools.AxisPart.POSITIVE);
            Vector3 corner1 = DataBlockConverterTools.getNearest(out, vector, DataBlockConverterTools.Axis.Y, DataBlockConverterTools.AxisPart.POSITIVE);
            out.clear();
            DataBlockConverterTools.fillVectors(vectors, vector, out, DataBlockConverterTools.Axis.Y, DataBlockConverterTools.AxisPart.POSITIVE);
            Vector3 corner2 = DataBlockConverterTools.getNearest(out, vector, DataBlockConverterTools.Axis.X, DataBlockConverterTools.AxisPart.POSITIVE);
            getLines(primitives, vector, corner1, corner2);
        }

        return primitives;
    }
}
