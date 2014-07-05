package ogla.axischart2D.lists;

import ogla.axischart.lists.BaseList;
import ogla.axischart2D.AxisChart2D;
import ogla.axischart2D.plugin.chartcomponent.AxisChart2DComponent;

public final class ListAxisChart2DInstalledComponent extends BaseList<AxisChart2DComponent> {
    protected AxisChart2D axisChart;

    public ListAxisChart2DInstalledComponent(AxisChart2D axisChart) {
        super();
        this.axisChart = axisChart;
    }

    @Override
    public boolean add(AxisChart2DComponent axisChartComponent) {
        boolean b = super.add(axisChartComponent);
        if (b) {
            axisChartComponent.attachTo(axisChart);
        }
        return b;
    }

    @Override
    public int remove(AxisChart2DComponent chartComponent) {
        int i = super.remove(chartComponent);
        if (i != -1) {
            chartComponent.attachTo(null);
        }
        return i;
    }

    @Override
    public AxisChart2DComponent remove(int i) {
        AxisChart2DComponent chartComponent = super.remove(i);
        if (chartComponent != null) {
            chartComponent.attachTo(null);
        }
        return chartComponent;
    }
}
