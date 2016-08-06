
package ogla.datafitting.osgi;

import ogla.exdata.DataBundle;
import ogla.datafitting.Init;
import org.osgi.framework.BundleActivator;
import org.osgi.framework.BundleContext;
import org.osgi.framework.ServiceReference;
import org.osgi.util.tracker.ServiceTracker;
import org.osgi.util.tracker.ServiceTrackerCustomizer;

/**
 *
 * @author Marcin
 */
public class Activator implements BundleActivator {

    private ServiceTracker bundleDataTracker;
    private Init init = new Init();
    private BundleContext bundleContext;

    private class BundleDataCustomizer implements ServiceTrackerCustomizer {

        public Object addingService(ServiceReference sr) {
            DataBundle bundleData = (DataBundle) bundleContext.getService(sr);
            init.add(bundleData);
            return bundleData;
        }

        public void modifiedService(ServiceReference sr, Object o) {
        }

        public void removedService(ServiceReference sr, Object o) {
            DataBundle bundleData = (DataBundle) o;
            init.remove(bundleData);
        }
    }

    public void start(BundleContext context) {
        bundleContext = context;
        DataBundle bundleData = init.getBundleData();
        context.registerService(DataBundle.class.getName(), bundleData, null);
        bundleDataTracker = new ServiceTracker(context, DataBundle.class.getName(), new BundleDataCustomizer());
        bundleDataTracker.open();
    }

    public void stop(BundleContext context) {
        bundleDataTracker.close();
    }
}
