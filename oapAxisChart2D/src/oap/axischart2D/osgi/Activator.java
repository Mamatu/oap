package ogla.axischart2D.osgi;

import java.util.List;
import ogla.axischart.AxisChartBundle;
import ogla.axischart.lists.BaseList;
import ogla.axischart.osgi.BaseActivator;
import ogla.axischart2D.AxisChart2DBundle;
import ogla.axischart2D.lists.AxisChart2DComponentBundleList;
import ogla.axischart2D.lists.PlotManagerBundleList;
import ogla.chart.Chart;
import ogla.chart.ChartObject;
import ogla.chart.ChartSurface;
import org.osgi.framework.BundleContext;

public class Activator extends BaseActivator {

    protected AxisChart2DComponentBundleList chartComponentBundleList = new AxisChart2DComponentBundleList();
    protected PlotManagerBundleList plotManagerBundleList = new PlotManagerBundleList();
    AxisChart2DBundle axisChart2DBundle = null;

    @Override
    public void beforeStart(BundleContext bc, List<BaseList> baseLists) {
    }

    @Override
    public void afterStart(BundleContext bc, List<BaseList> baseLists) {
    }

    @Override
    public void beforeOpen(BundleContext bc, List<BaseList> baseLists) {
        baseLists.add(chartComponentBundleList);
        baseLists.add(plotManagerBundleList);
    }

    @Override
    public void afterOpen(BundleContext bc, List<BaseList> baseLists) {
    }

    @Override
    protected AxisChartBundle[] newAxisChartBundles(List<BaseList> baseLists) {
        AxisChartBundle[] array = {new AxisChart2DBundle(baseLists)};
        return array;
    }
}
