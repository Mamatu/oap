package ogla.axischart3D.plugin.plotmanager;

import ogla.data.DataBlock;

import ogla.g3d.Line3D;
import ogla.g3d.Primitive;
import ogla.g3d.Triangle3D;

import ogla.math.Vector3;
import java.util.ArrayList;
import java.util.List;

public class DataBlockConverterToModel extends DataBlockConverter {

    private Triangle3D isSame(Vector3 v1, Vector3 v2, Vector3 v3, List<Primitive> primitives) {
        if (v1 == null || v2 == null || v3 == null) {
            return null;
        }
        for (Primitive primitive : primitives) {
            Triangle3D triangle3D = (Triangle3D) primitive;
            if (triangle3D.compare(v1, v2, v3)) {
                return null;
            }
        }
        return new Triangle3D(v1, v2, v3);
    }

    private void addTo(Triangle3D triangle3D, List<Primitive> primitives) {
        if (triangle3D != null) {
            primitives.add(triangle3D);
        }
    }

    public Triangle3D getTriangles(List<Primitive> primitives, Vector3 refCorner, Vector3 corner1, Vector3 corner2) {
        Triangle3D triangle3D = isSame(refCorner, corner1, corner2, primitives);
        addTo(triangle3D, primitives);
        return triangle3D;
    }

    private Vector3 getVector(DataBlock dataBlock, int row) {
        float x = dataBlock.get(row, 0).getNumber().floatValue();
        float y = dataBlock.get(row, 1).getNumber().floatValue();
        float z = dataBlock.get(row, dataBlock.columns() - 1).getNumber().floatValue();
        return new Vector3(x, y, z);
    }

    @Override
    public List<Primitive> getPrimitives(DataBlock dataBlock) {
        Vector3 average = new Vector3();
        List<Primitive> primitives = new ArrayList<Primitive>();
        List<Vector3> vectors = new ArrayList<Vector3>();
        for (int fa = 0; fa < dataBlock.rows(); fa++) {
            Vector3 vec = getVector(dataBlock, fa);
            vectors.add(vec);
        }
        List<Vector3> out = new ArrayList<Vector3>();
     
        for (int fa = 0; fa < vectors.size(); fa++) {
            Vector3 vector = vectors.get(fa);

            DataBlockConverterTools.fillVectors(vectors, vector, out, DataBlockConverterTools.Axis.X, DataBlockConverterTools.AxisPart.POSITIVE);
            Vector3 corner1 = out.get(0);//DataBlockConverterTools.getNearest(out, vector, DataBlockConverterTools.Axis.Y, DataBlockConverterTools.AxisPart.POSITIVE);
            out.clear();
            DataBlockConverterTools.fillVectors(vectors, vector, out, DataBlockConverterTools.Axis.Y, DataBlockConverterTools.AxisPart.POSITIVE);
            Vector3 corner2 = out.get(0);//DataBlockConverterTools.getNearest(out, vector, DataBlockConverterTools.Axis.X, DataBlockConverterTools.AxisPart.POSITIVE);
            getTriangles(primitives, vector, corner1, corner2);
            out.clear();

            DataBlockConverterTools.fillVectors(vectors, vector, out, DataBlockConverterTools.Axis.X, DataBlockConverterTools.AxisPart.NEGATIVE);
            corner1 = out.get(0);//DataBlockConverterTools.getNearest(out, vector, DataBlockConverterTools.Axis.Y, DataBlockConverterTools.AxisPart.NEGATIVE);
            out.clear();
            DataBlockConverterTools.fillVectors(vectors, vector, out, DataBlockConverterTools.Axis.Y, DataBlockConverterTools.AxisPart.NEGATIVE);
            corner2 = out.get(0);//DataBlockConverterTools.getNearest(out, vector, DataBlockConverterTools.Axis.X, DataBlockConverterTools.AxisPart.NEGATIVE);
            getTriangles(primitives, vector, corner1, corner2);
            out.clear();
        }

        return primitives;
    }
}
