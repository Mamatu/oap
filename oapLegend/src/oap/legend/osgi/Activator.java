package ogla.legend.osgi;

import ogla.axischart.plugin.image.ImageComponentBundle;
import ogla.legend.BundleLegend;
import org.osgi.framework.BundleActivator;
import org.osgi.framework.BundleContext;

public class Activator implements BundleActivator {

    private BundleLegend bundleLabel = new BundleLegend();

    public void start(BundleContext bc) throws Exception {
        bc.registerService(ImageComponentBundle.class.getName(), bundleLabel, null);

    }

    public void stop(BundleContext bc) throws Exception {
    }
}
