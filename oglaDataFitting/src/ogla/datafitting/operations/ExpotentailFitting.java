package ogla.datafitting.operations;

import ogla.exdata.DataBundle;
import ogla.exdata.DataBlock;
import ogla.exdata.DataRepository;
import ogla.datafitting.Operation;

public class ExpotentailFitting extends Operation {

    public ExpotentailFitting(DataBundle bundleData) {
        super(bundleData);
    }
    private Math math;

    private float[] c(DataBlock dataRepository, int ax, int ay) {
        float sly = 0.f;
        float sx2 = 0.f;
        float sx = 0.f;
        float sxly = 0.f;
        float s2x = 0.f;
        for (int fa = 0; fa < dataRepository.rows(); fa++) {
            float x = dataRepository.get(fa, ax).getNumber().floatValue();
            float y = dataRepository.get(fa, ay).getNumber().floatValue();
            float ly = (float) math.log((double) y);
            sly += ly;
            sx2 += x * x;
            sx += x;
            sxly += x * ly;
        }
        s2x = sx * sx;
        float a = sly * sx2 - sx * sxly;
        a = a / ((float) dataRepository.rows() * sx2 - s2x);
        float b = (float) dataRepository.rows() * sxly - sx * sly;
        b = b / ((float) dataRepository.rows() * sx2 - s2x);
        a = (float) math.exp(a);
        float[] c = {a, b};
        return c;
    }

    @Override
    public DataBlock modify(DataBlock dataBlock, DataRepositoryImpl dataRepository, float begin, float end, float step, float degree,
            int ax, int ay) {
        DataBlockImpl dataBlockImpl = new DataBlockImpl(dataRepository);


        float[] c = this.c(dataBlock, ax, ay);
        float current = begin;
        while (current <= end) {
            float x = current;
            float y = c[0] * (float) math.exp(c[1] * x);
            dataBlockImpl.rows.add(new DataRowImpl(x, y));
            current += step;
        }
        return dataBlockImpl;
    }

    @Override
    public String getLabel() {
        return "Expotentail fitting";
    }

    @Override
    public String getSuffix() {
        return "ef";
    }
}
