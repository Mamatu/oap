package ogla.grid.osgi;

import ogla.axischart2D.plugin.chartcomponent.AxisChart2DComponentBundle;
import ogla.grid.Grid;
import ogla.grid.GridHorizontal;
import ogla.grid.GridVertical;
import org.osgi.framework.BundleActivator;
import org.osgi.framework.BundleContext;

/**
 *
 * @author marcin
 */
public class Activator implements BundleActivator {

    protected Grid.AxisChart2DComponentBundleImpl gridBundle = new Grid.AxisChart2DComponentBundleImpl();
    protected GridHorizontal.AxisChart2DComponentBundleImpl gridHorizontalBundle = new GridHorizontal.AxisChart2DComponentBundleImpl();
    protected GridVertical.AxisChart2DComponentBundleImpl gridVerticalBundle = new GridVertical.AxisChart2DComponentBundleImpl();
    protected Grid grid = new Grid(gridBundle, gridBundle.getIndex());
    protected GridHorizontal gridHorizontal = new GridHorizontal(gridHorizontalBundle, gridHorizontalBundle.getIndex());
    protected GridVertical gridVertical = new GridVertical(gridVerticalBundle, gridVerticalBundle.getIndex());

    public void start(BundleContext bc) {
        bc.registerService(AxisChart2DComponentBundle.class.getName(), grid.getBundle(), null);
        bc.registerService(AxisChart2DComponentBundle.class.getName(), gridHorizontal.getBundle(), null);
        bc.registerService(AxisChart2DComponentBundle.class.getName(), gridVertical.getBundle(), null);
    }

    public void stop(BundleContext bc) {
    }
}
