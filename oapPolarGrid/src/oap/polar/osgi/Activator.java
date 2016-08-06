package ogla.polar.osgi;

import ogla.axischart2D.plugin.chartcomponent.AxisChart2DComponentBundle;
import ogla.polar.BundlePolarGrid;
import org.osgi.framework.BundleActivator;
import org.osgi.framework.BundleContext;

public class Activator implements BundleActivator {

    protected BundlePolarGrid bundlePolarGrid = new BundlePolarGrid();

    public void start(BundleContext bc) {
        bc.registerService(AxisChart2DComponentBundle.class.getName(), bundlePolarGrid, null);
    }

    public void stop(BundleContext bc) {
    }
}
