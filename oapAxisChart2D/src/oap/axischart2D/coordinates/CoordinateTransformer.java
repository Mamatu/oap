
package ogla.axischart2D.coordinates;

import ogla.chart.Chart;
import ogla.axischart2D.AxisChart2D;

/**
 * Transforme chart data into screen coordinates.
 */
public interface CoordinateTransformer {

    /**
     * Transforme chart data into screen coordinates.
     * @param chart on which we prepare transformation.
     * @param value chart's data
     * @return position this unit on panel
     */
    public Number transform(AxisChart2D chart, Number value);


  
    /**
     * Check if value is in scope (can be displayed on chart).
     * @param chart on which we prepare transformation.
     * @param value Number
     * @return true - in range, false - otherwise
     */
    public boolean inScope(AxisChart2D chart, Number value);

    /**
     * Check if position on chart surface is range (can be displayed on chart).
     * @param chart on which we prepare transformation.
     * @param position
     * @return
     */
    public boolean inRange(Chart chart, Number position);

    /**
     * Transform screen coordinates into chart data.
     * @param chart on which we prepare transformation.
     * @param positions positions of units
     * @return array of chart's data
     */
    public Number[] retransform(AxisChart2D chart, Number[] positions);
}
