package ogla.axischart2D.coordinates;

import ogla.chart.Chart;
import ogla.axischart2D.AxisChart2D;

/**
 *Transformed value into Y screen coordinates.
 * 
 */
public class CoordinateYAxis implements CoordinateTransformer {

    public Number transform(AxisChart2D chart, Number value) {
        double fvalue = value.doubleValue();
        double low = chart.getYTicks().getLowestTick().doubleValue();
        double great = chart.getYTicks().getGreatestTick().doubleValue();
        fvalue = fvalue - low;
        double lg = great - low;
        int height = chart.getHeight();
        int out = height - (int) ((fvalue * height) / lg);
        out = out + chart.getLocation().y;
        return out;
    }

    public boolean inScope(AxisChart2D chart, Number value) {
        if (value.floatValue() < chart.getYTicks().getLowestTick().floatValue() || value.floatValue() > chart.getYTicks().getGreatestTick().floatValue()) {
            return false;
        }
        return true;
    }

    public Number[] retransform(AxisChart2D chart, Number[] positions) {
        double low = chart.getYTicks().getLowestTick().doubleValue();
        double great = chart.getYTicks().getGreatestTick().doubleValue();
        double position = positions[0].doubleValue();
        double lg = great - low;
        int height = chart.getHeight();
        position = position - chart.getLocation().y;
        double fvalue = ((height - position) * lg) / height;
        fvalue = fvalue + low;
        Number[] numbers = {fvalue};
        return numbers;

    }

    public boolean inRange(Chart chart, Number position) {
        int pos = position.intValue();
        if (pos >= chart.getLocation().y && pos <= chart.getHeightEndOnSurface()) {
            return true;
        }
        return false;
    }
}
