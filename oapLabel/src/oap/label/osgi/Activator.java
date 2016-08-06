
package ogla.label.osgi;

import ogla.axischart.plugin.image.ImageComponentBundle;
import ogla.label.BundleLabel;
import org.osgi.framework.BundleActivator;
import org.osgi.framework.BundleContext;


public class Activator implements BundleActivator {

    private BundleLabel bundleLabel = new BundleLabel();

    public void start(BundleContext bc) throws Exception {
        bc.registerService(ImageComponentBundle.class.getName(), bundleLabel, null);

    }

    public void stop(BundleContext bc) throws Exception {
    }
}
