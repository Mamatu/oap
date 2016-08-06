package ogla.axis;

import ogla.axischart2D.AxisChart2D;
import ogla.axischart2D.plugin.chartcomponent.AxisChart2DComponent;
import ogla.axischart2D.plugin.chartcomponent.AxisChart2DComponentBundle;
import java.awt.Color;

public abstract class AxisComponent extends AxisChart2DComponent {

    protected boolean isChange = false;
    protected int alpha = 100;
    protected Color color;
    protected AxisChart2D axisChart2D;

    public AxisChart2D getAxisChart2D() {
        return axisChart2D;
    }

    public AxisComponent(AxisChart2DComponentBundle bundleChartComponent) {
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

    public void attachTo(AxisChart2D axisChart2D) {
        this.axisChart2D = axisChart2D;
    }
}
