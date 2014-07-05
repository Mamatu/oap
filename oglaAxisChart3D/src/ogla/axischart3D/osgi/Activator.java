package ogla.axischart3D.osgi;

import ogla.axischart.AxisChartBundle;
import ogla.axischart.lists.BaseList;
import ogla.axischart3D.AxisChart3DBundle;
import ogla.axischart.osgi.BaseActivator;
import ogla.axischart3D.lists.ListAxisChart3DComponentBundle;
import ogla.axischart3D.lists.ListAxisChart3DPlotManagerBundle;
import ogla.axischart3D.lists.ListProjectionBundle;
import java.util.List;
import org.osgi.framework.BundleContext;

public class Activator extends BaseActivator {

    private ListAxisChart3DComponentBundle listAxisChart3DComponentBundle = new ListAxisChart3DComponentBundle();
    private ListAxisChart3DPlotManagerBundle listAxisChart3DPlotManagerBundle = new ListAxisChart3DPlotManagerBundle();
    private ListProjectionBundle listProjectionBundle = new ListProjectionBundle();

    @Override
    public void beforeStart(BundleContext bc, List<BaseList> baseLists) {
    }

    @Override
    public void beforeOpen(BundleContext bc, List<BaseList> baseLists) {
        baseLists.add(listAxisChart3DComponentBundle);
        baseLists.add(listAxisChart3DPlotManagerBundle);
        baseLists.add(listProjectionBundle);
    }

    @Override
    public void afterStart(BundleContext bc, List<BaseList> baseLists) {
    }

    @Override
    public void afterOpen(BundleContext bc, List<BaseList> baseLists) {
    }

    @Override
    protected AxisChartBundle[] newAxisChartBundles(List<BaseList> baseLists) {
        AxisChartBundle[] array = {new AxisChart3DBundle(baseLists)};
        return array;
    }
}
