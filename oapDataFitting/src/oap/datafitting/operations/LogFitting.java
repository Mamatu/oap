package ogla.datafitting.operations;

import ogla.exdata.DataBundle;
import ogla.exdata.DataBlock;
import ogla.exdata.DataRepository;
import ogla.datafitting.Operation;
import java.util.Collections;

/**
 *
 * @author marcin
 */
public class LogFitting extends Operation {

    public LogFitting(DataBundle bundleData) {
        super(bundleData);
    }
    private Math math;

    private float[] c(DataBlock repositoryData, int ax, int ay) {
        float sy = 0.f;
        float slx = 0.f;
        float sl2x = 0.f;
        float sylx = 0.f;
        for (int fa = 0; fa < repositoryData.rows();
                fa++) {
            float x = repositoryData.get(fa, ax).getNumber().floatValue();
            if (x > 0.f) {
                float y = repositoryData.get(fa, ay).getNumber().floatValue();
                float lx = (float) math.log(x);

                sy += y;
                slx += lx;
                sylx += y * lx;
                sl2x += lx * lx;
            }
        }
        float b = ((float) repositoryData.rows()) * sylx - sy * slx;
        b = b / (((float) repositoryData.rows()) * sl2x - slx * slx);

        float a = (sy - b * slx);
        a = a / ((float) repositoryData.rows());
        float[] c = {a, b};
        return c;

    }

    @Override
    public DataBlock modify(DataBlock dataBlock, DataRepositoryImpl dataRepository,float begin, float end, float step, float degree, int ax, int ay) {
        DataBlockImpl repositoryData1 = new DataBlockImpl(dataRepository);
       
        float[] c = this.c(dataBlock, ax, ay);

        float current = begin;
        while (current <= end) {
            if (current > 0.f) {
                float x = current;
                float y = c[0] + (float) math.log(x) * c[1];
                repositoryData1.rows.add(new DataRowImpl(x, y));
            }
            current += step;
        }
        Collections.sort(repositoryData1.rows);
        return repositoryData1;
    }

    @Override
    public String getLabel() {
        return "Log fitting";
    }

    @Override
    public String getSuffix() {
        return "lf";
    }
}
