package ogla.axischart.osgi;

import java.lang.reflect.ParameterizedType;
import java.util.ArrayList;
import java.util.List;
import ogla.axischart.AxisChartBundle;
import ogla.axischart.lists.BaseList;
import ogla.axischart.lists.ListDataBundle;
import ogla.axischart.lists.ListImageComponentBundle;
import ogla.axischart.lists.ListImageExporterBundle;
import ogla.core.application.ApplicationBundle;
import org.osgi.framework.BundleActivator;
import org.osgi.framework.BundleContext;
import org.osgi.framework.ServiceReference;
import org.osgi.util.tracker.ServiceTracker;
import org.osgi.util.tracker.ServiceTrackerCustomizer;

public abstract class BaseActivator implements BundleActivator {

    protected List<ServiceTracker> serviceTrackers = new ArrayList<ServiceTracker>();

    protected final ServiceTracker newOpenedServiceTracker(final BaseList baseList, final BundleContext bc) {
        ParameterizedType pt = (ParameterizedType) baseList.getClass().getGenericSuperclass();

        String name = pt.getActualTypeArguments()[0].toString();
        name = name.split(" ")[1];

        ServiceTrackerCustomizer serviceTrackerCustomizer = new ServiceTrackerCustomizer() {

            public Object addingService(ServiceReference sr) {
                Object obj = bc.getService(sr);
                baseList.add(obj);
                return obj;
            }

            public void modifiedService(ServiceReference sr, Object o) {
            }

            public void removedService(ServiceReference sr, Object o) {
                baseList.remove(o);
            }
        };
        ServiceTracker tracker = new ServiceTracker(bc, name, serviceTrackerCustomizer);
        tracker.open();
        serviceTrackers.add(tracker);
        return tracker;
    }

    private void closeAllServiceTrackers() {
        for (ServiceTracker serviceTracker : serviceTrackers) {
            serviceTracker.close();
        }
        serviceTrackers.clear();
    }
    protected ListDataBundle listDataBundle = new ListDataBundle();
    protected ListImageComponentBundle listImageComponentBundle = new ListImageComponentBundle();
    protected ListImageExporterBundle listImageExporterBundle = new ListImageExporterBundle();

    public abstract void beforeStart(BundleContext bc, List<BaseList> baseLists);

    public abstract void beforeOpen(BundleContext bc, List<BaseList> baseLists);

    public abstract void afterStart(BundleContext bc, List<BaseList> baseLists);

    public abstract void afterOpen(BundleContext bc, List<BaseList> baseLists);

    protected abstract AxisChartBundle[] newAxisChartBundles(List<BaseList> baseLists);

    public void start(BundleContext bc) throws Exception {
        List<BaseList> baseLists = new ArrayList<BaseList>();
        baseLists.add(listDataBundle);
        baseLists.add(listImageComponentBundle);
        baseLists.add(listImageExporterBundle);

        beforeOpen(bc, baseLists);

        for (BaseList baseList : baseLists) {
            this.newOpenedServiceTracker(baseList, bc);
        }

        afterOpen(bc, baseLists);

        beforeStart(bc, baseLists);

        for (AxisChartBundle axisChartBundle : newAxisChartBundles(baseLists)) {
            bc.registerService(ApplicationBundle.class.getName(), axisChartBundle, null);
        }
        afterStart(bc, baseLists);
    }

    public void stop(BundleContext bc) throws Exception {
        closeAllServiceTrackers();
    }
}
