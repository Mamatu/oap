package ogla.axischart2D.coordinates;

import ogla.chart.Chart;
import ogla.axischart2D.AxisChart2D;

/**
 * Transformed chart data into X screen coordinates.
 * 
 */
public class CoordinateXAxis implements CoordinateTransformer {

    public boolean inScope(AxisChart2D chart, Number value) {
        if (value.floatValue() < chart.getXTicks().getLowestTick().floatValue() || value.floatValue() > chart.getXTicks().getGreatestTick().floatValue()) {
            return false;
        }
        return true;
    }

    public Number transform(AxisChart2D chart, Number value) {
        double fvalue = value.doubleValue();
        double low = chart.getXTicks().getLowestTick().doubleValue();
        double great = chart.getXTicks().getGreatestTick().doubleValue();
        fvalue = fvalue - low;
        double lg = great - low;
        int width = chart.getWidth();
        int out = (int) ((fvalue * width) / lg);
        out = out + chart.getLocation().x;
        return out;
    }

    public Number[] retransform(AxisChart2D chart, Number[] positions) {
        double low = chart.getXTicks().getLowestTick().doubleValue();
        double great = chart.getXTicks().getGreatestTick().doubleValue();
        int width = chart.getWidth();
        double position = positions[0].doubleValue();
        position = position - chart.getLocation().x;
        double lg = great - low;
        double fvalue = (position * lg) / width;
        fvalue = fvalue + low;
        Number[] numbers = {fvalue};
        return numbers;
    }

    public boolean inRange(Chart chart, Number position) {
        int pos = position.intValue();
        if (pos >= chart.getLocation().x && pos <= chart.getWidthEndOnSurface()) {
            return true;
        }
        return false;
    }
}
