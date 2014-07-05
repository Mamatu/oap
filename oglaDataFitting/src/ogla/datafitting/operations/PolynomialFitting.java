package ogla.datafitting.operations;

import ogla.exdata.DataBundle;
import ogla.exdata.DataBlock;
import ogla.exdata.DataRepository;
import ogla.datafitting.Operation;
import ogla.math.Matrix;

/**
 *
 * @author marcin
 */
public class PolynomialFitting extends Operation {

    public PolynomialFitting(DataBundle bundleData) {
        super(bundleData);
    }

    @Override
    public String getSuffix() {
        return "pf";
    }

    @Override
    public DataBlock modify(DataBlock dataBlock, DataRepositoryImpl dataRepository, float begin, float end, float step, float degree, int ax, int ay) {
        DataBlockImpl repositoryData1 = new DataBlockImpl(dataRepository);

        Matrix m = new Matrix(dataBlock.rows(), (int)degree);
        Matrix ys = new Matrix(dataBlock.rows(), 1);
        for (int fa = 0; fa < dataBlock.rows(); fa++) {
            m.set(1.f, fa, 0);

            ys.set(dataBlock.get(fa, ay).getNumber().floatValue(), fa, 0);
            for (int fb = 1; fb < degree; fb++) {
                m.set((float) Math.pow(dataBlock.get(fa, ax).getNumber().floatValue(), fb), fa, fb);
            }
        }
        Matrix t = m.transpose();
        Matrix i = t.multiply(m);
        i = i.inverse();
        Matrix a = i.multiply(t).multiply(ys);

        float current = begin;
        while (current <= end) {
            float y = 0.f;
            for (int fa = 0; fa < a.rows(); fa++) {
                y += a.getFloat(fa, 0) * (float) Math.pow(current, fa);
            }
            repositoryData1.rows.add(new DataRowImpl(current, y));
            current += step;
        }
        return repositoryData1;
    }

    @Override
    public String getLabel() {
        return "Polynomial fitting";
    }
}
