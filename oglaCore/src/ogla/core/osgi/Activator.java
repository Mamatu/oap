/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.core.osgi;

import java.util.logging.Level;
import java.util.logging.Logger;
import ogla.core.gui.PluginsFrame;
import ogla.core.gui.PluginsManager;
import ogla.core.gui.PluginsTree;
import ogla.core.ui.Interactor;
import org.osgi.framework.BundleActivator;
import org.osgi.framework.BundleContext;
import org.osgi.framework.BundleException;
import org.osgi.framework.ServiceReference;
import org.osgi.util.tracker.ServiceTracker;
import org.osgi.util.tracker.ServiceTrackerCustomizer;

public class Activator implements BundleActivator {

    private ServiceTracker serviceTracker = null;
    private PluginsFrame pluginsFrame = null;

    @Override
    public void start(final BundleContext bc) throws BundleException {
        try {
            pluginsFrame = new PluginsFrame(bc, new PluginsManager(bc), new PluginsTree());
            pluginsFrame.setVisible(true);
            this.serviceTracker = new ServiceTracker(bc, Interactor.class.getName(), new ServiceTrackerCustomizer() {
                @Override
                public Object addingService(ServiceReference sr) {
                    Object object = bc.getService(sr);
                    Interactor interactor = (Interactor) object;
                    interactor.start();
                    return object;
                }

                @Override
                public void modifiedService(ServiceReference sr, Object o) {
                }

                @Override
                public void removedService(ServiceReference sr, Object o) {
                    if (o instanceof Interactor) {
                        Interactor interactor = (Interactor) o;
                        interactor.stop();
                    }
                }
            });
            this.serviceTracker.open();
        } catch (Exception ex) {
            Logger.getLogger(Activator.class.getName()).log(Level.SEVERE, null, ex);
            throw ex;
        }
    }

    @Override
    public void stop(BundleContext bc) throws BundleException {
        pluginsFrame = null;
    }
}
