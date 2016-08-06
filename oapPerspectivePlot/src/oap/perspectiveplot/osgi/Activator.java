package ogla.perspectiveplot.osgi;


import ogla.axischart3D.plugin.plotmanager.AxisChart3DPlotManagerBundle;
import ogla.axischart3D.plugin.projection.ProjectionBundle;
import ogla.perspectiveplot.PerspectiveBundle;
import org.osgi.framework.BundleActivator;
import org.osgi.framework.BundleContext;

public class Activator implements BundleActivator {

    protected PerspectiveBundle perspectivePlotBundle = new PerspectiveBundle();

    public void start(BundleContext context) throws Exception {
        context.registerService(ProjectionBundle.class.getName(), perspectivePlotBundle,
                null);
    }

    public void stop(BundleContext context) throws Exception {
    }
}
