
package ogla.axis.util;

import ogla.core.data.DataValue;
import ogla.axischart2D.AxisChart2D;
import ogla.axischart2D.Ticks;
import ogla.axischart2D.coordinates.CoordinateTransformer;
import java.util.ArrayList;
import java.util.List;


public class LocationManager {

    protected Ticks ticks;
    protected CoordinateTransformer coordinateTransformer;
    protected AxisChart2D axisChart;
    protected List<DataValue> list = new ArrayList<DataValue>();

    public LocationManager(AxisChart2D axisChart, Ticks ticks, CoordinateTransformer coordinateTransformer) {
        this.ticks = ticks;
        this.axisChart = axisChart;
        this.coordinateTransformer = coordinateTransformer;
    }

    public DataValue nearestTick(int position) {
        Number[] positions = {position};
        Number value = coordinateTransformer.retransform(axisChart, positions)[0];
        DataValue valueData = null;
        float diff = 0;
        if (ticks.getSizeOfList() >= 1) {
            valueData = ticks.getDataValue(0);
            diff = value.floatValue() - ticks.getDataValue(0).getNumber().floatValue();
            if (diff < 0) {
                diff = -diff;
            }
        }
        for (int fa = 1; fa < ticks.getSizeOfList(); fa++) {
            float tdiff = value.floatValue() - ticks.getDataValue(fa).getNumber().floatValue();
            if (tdiff < 0) {
                tdiff = -tdiff;
            }
            if (tdiff < diff) {
                valueData = ticks.getDataValue(fa);
                diff = tdiff;
            }
        }
        return valueData;
    }
}
