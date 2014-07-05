package ogla.datafitting.operations;

import ogla.exdata.DataBundle;
import ogla.exdata.DataBlock;
import ogla.exdata.DataRepository;
import ogla.datafitting.Operation;

public class GaussianFitting extends Operation {

    public GaussianFitting(DataBundle dataBundle) {
        super(dataBundle);
    }
    private float magicNumber = (float) Math.sqrt(2.d * Math.PI);
    private float e = 2.71828183f;

    private float normalDistribution(float x, float location, float deviation) {
        float out = 1.f / (magicNumber * deviation);
        float exponent = -(x - location) * (x - location) / (2 * deviation * deviation);
        if (exponent > -1.f && exponent < 1.f) {
            out = out * (float) Math.exp(exponent);
        } else {
            out = out * (float) Math.exp(exponent);
        }
        return out;
    }

    @Override
    public DataBlock modify(DataBlock dataBlock, DataRepositoryImpl dataRepository, float begin, float end, float step, float degree, int ax, int ay) {
        

        boolean isAmplitude = false;
        float amplitude = degree;
        float location = 0;
        float deviation = 0;
        float sumys = 0;
        int c = 0;
        for (int fa = 0; fa < dataBlock.rows(); fa++) {
            float v = dataBlock.get(fa, ax).getNumber().floatValue();
            location += v * dataBlock.get(fa, 1).getNumber().floatValue();
            sumys += dataBlock.get(fa, ay).getNumber().floatValue();
            if (!isAmplitude && amplitude < dataBlock.get(fa, 1).getNumber().floatValue()) {
                amplitude = dataBlock.get(fa, ay).getNumber().floatValue();
            }
        }


        location = location / sumys;
        for (int fa = 0; fa < dataBlock.rows(); fa++) {
            float v = dataBlock.get(fa, 0).getNumber().floatValue();
            deviation += (v - location) * (v - location) * dataBlock.get(fa, 1).getNumber().floatValue();
        }
        deviation = deviation / sumys;
        deviation = (float) Math.sqrt((double) deviation);

        Operation.DataBlockImpl dataRepositoryImpl = new Operation.DataBlockImpl(dataRepository);
        float x = begin;
        while (x < end) {
            float y = amplitude * normalDistribution(x, location, deviation);
            Operation.DataRowImpl dataRowImpl = new Operation.DataRowImpl(x, y);
            dataRepositoryImpl.rows.add(dataRowImpl);
            x += step;
        }

        return dataRepositoryImpl;
    }

    @Override
    public String getLabel() {
        return "GaussianFitting";
    }

    @Override
    public String getSuffix() {
        return "gf";
    }
}
