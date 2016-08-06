
package ogla.plot.errorpoint.osgi;


import ogla.axischart2D.plugin.plotmanager.AxisChart2DPlotManagerBundle;
import ogla.plot.errorpoint.PlotErrorPointManagerFactory;
import org.osgi.framework.BundleActivator;
import org.osgi.framework.BundleContext;

public class Activator implements BundleActivator {

    AxisChart2DPlotManagerBundle plotManagerFactory = null;

    public void start(BundleContext bc) throws Exception {
        plotManagerFactory = new PlotErrorPointManagerFactory();
        bc.registerService(AxisChart2DPlotManagerBundle.class.getName(), plotManagerFactory, null);
    }

    public void stop(BundleContext bc) throws Exception {
    }
}
    
