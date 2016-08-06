
package ogla.plot.bar.osgi;


import ogla.axischart2D.plugin.plotmanager.AxisChart2DPlotManagerBundle;
import ogla.plot.bar.PlotBarManagerBundle;
import org.osgi.framework.BundleActivator;
import org.osgi.framework.BundleContext;

public class Activator implements BundleActivator {

    AxisChart2DPlotManagerBundle plotManagerBundle = null;

    public void start(BundleContext bc) throws Exception {
        plotManagerBundle = new PlotBarManagerBundle();
        bc.registerService(AxisChart2DPlotManagerBundle.class.getName(), plotManagerBundle, null);
    }

    public void stop(BundleContext bc) throws Exception {
    }
}
    