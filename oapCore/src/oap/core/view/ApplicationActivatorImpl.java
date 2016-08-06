package ogla.core.view;

import java.util.ArrayList;
import java.util.List;
import ogla.core.application.ApplicationBundle;
import org.osgi.framework.BundleActivator;
import org.osgi.framework.BundleContext;
import org.osgi.framework.ServiceReference;
import org.osgi.util.tracker.ServiceTracker;
import org.osgi.util.tracker.ServiceTrackerCustomizer;

/**
 * Default class which provide
 */
public abstract class ApplicationActivatorImpl implements BundleActivator {

    protected ServiceTracker trackerApplication;
    protected BundleContext bundleContext;

    public static interface Listener {

        public void wasAddedApplication(ApplicationBundle applicationBundle);

        public void wasRemovedApplication(ApplicationBundle applicationBundle);
    }
    private List<Listener> listeners = new ArrayList<Listener>();

    private void addedApplicationBundle(ApplicationBundle applicationBundle) {
        for (Listener listener : listeners) {
            listener.wasAddedApplication(applicationBundle);
        }
    }

    private void removedApplicationBundle(ApplicationBundle applicationBundle) {
        for (Listener listener : listeners) {
            listener.wasRemovedApplication(applicationBundle);
        }
    }

    public void addListener(Listener listener) {
        listeners.add(listener);
    }

    public void removeListener(Listener listener) {
        listeners.remove(listener);
    }

    public void start(BundleContext bundleContext) {
        beforeStart(bundleContext);
        this.bundleContext = bundleContext;
        trackerApplication = new ServiceTracker(bundleContext, ApplicationBundle.class.getName(), new ApplicationTracker());
        trackerApplication.open();
        startImpl(bundleContext);
    }

    public abstract void beforeStart(BundleContext bundleContext);

    public abstract void startImpl(BundleContext bundleContext);

    public void stop(BundleContext bundleContext) {
        trackerApplication.close();
        stopImpl(bundleContext);
    }

    public abstract void stopImpl(BundleContext bundleContext);

    private class ApplicationTracker implements ServiceTrackerCustomizer {

        public Object addingService(ServiceReference reference) {
            ApplicationBundle applicationBUndle = (ApplicationBundle) bundleContext.getService(reference);
            ApplicationActivatorImpl.this.addedApplicationBundle(applicationBUndle);
            return applicationBUndle;
        }

        public void modifiedService(ServiceReference reference, Object object) {
        }

        @SuppressWarnings("unchecked")
        public void removedService(ServiceReference reference, Object object) {
            ApplicationBundle applicationBundle = (ApplicationBundle) object;
            ApplicationActivatorImpl.this.removedApplicationBundle(applicationBundle);
        }
    }
}
