
package ogla.grid;

import ogla.axischart2D.plugin.chartcomponent.AxisChart2DComponent;
import ogla.axischart2D.plugin.chartcomponent.AxisChart2DComponentBundle;
import java.awt.Color;

/**
 *
 * @author marcin
 */
public abstract class GridComponent extends AxisChart2DComponent {

    protected boolean isChange = false;
    protected int alpha = 100;
    protected Color color;

    public GridComponent(AxisChart2DComponentBundle bundleChartComponent) {
        super(bundleChartComponent);
    }

    public void isChange(boolean b) {
        isChange = b;
    }

    public void setAlpha(int alpha) {
        this.alpha = alpha;
    }

    public void setGlobalColor(Color color) {
        this.color = color;
    }
}
