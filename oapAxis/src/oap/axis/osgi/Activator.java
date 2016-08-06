package ogla.axis.osgi;

import ogla.axis.Axes;
import ogla.axis.AxisX;
import ogla.axis.AxisY;
import ogla.axischart2D.plugin.chartcomponent.AxisChart2DComponentBundle;
import org.osgi.framework.BundleActivator;
import org.osgi.framework.BundleContext;

public class Activator implements BundleActivator {

    protected Axes.AxisChart2DComponentBundleImpl axesBundle = new Axes.AxisChart2DComponentBundleImpl();
    protected AxisX.AxisChart2DComponentBundleImpl axisXBundle = new AxisX.AxisChart2DComponentBundleImpl();
    protected AxisY.AxisChart2DComponentBundleImpl axisYBundle = new AxisY.AxisChart2DComponentBundleImpl();
    protected Axes axes = new Axes(axesBundle, axesBundle.getIndex());
    protected AxisX axisX = new AxisX(axisXBundle, axisXBundle.getIndex());
    protected AxisY axisY = new AxisY(axisYBundle, axisYBundle.getIndex());

    public void start(BundleContext bc) {
        bc.registerService(AxisChart2DComponentBundle.class.getName(), axes.getBundle(), null);
        bc.registerService(AxisChart2DComponentBundle.class.getName(), axisX.getBundle(), null);
        bc.registerService(AxisChart2DComponentBundle.class.getName(), axisY.getBundle(), null);
    }

    public void stop(BundleContext bc) {
    }
}
