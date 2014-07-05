/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.groovy.osgi;

import java.util.HashMap;
import java.util.Map;
import ogla.core.application.ApplicationBundle;
import ogla.core.ui.Interactor;
import ogla.core.ui.OglaMethod;
import ogla.core.ui.OglaObject;
import ogla.core.view.ApplicationActivatorImpl;
import ogla.groovy.GroovyInteractor;
import ogla.groovy.InteractiveShell;
import org.osgi.framework.BundleContext;

/**
 *
 * @author mmatula
 */
public class Activator extends ApplicationActivatorImpl {

    private Map<OglaObject, ApplicationBundle> applicationsBundles = new HashMap<OglaObject, ApplicationBundle>();
    private GroovyInteractor groovyInteractor = new GroovyInteractor();

    private class ListenerImpl implements ApplicationActivatorImpl.Listener {

        @Override
        public void wasAddedApplication(ApplicationBundle applicationBundle) {
            OglaObject oglaObject = new OglaObject(applicationBundle.getSymbolicName(), null, null);
            applicationsBundles.put(oglaObject, applicationBundle);
            oglaObject.addMethod(new OglaMethod("create", null) {
                @Override
                public Object invoke(Object args, OglaObject oglaObject, Object userData) {
                    ApplicationBundle applicationBundle = Activator.this.applicationsBundles.get(oglaObject);
                    return applicationBundle.newApplication();
                }
            });
        }

        @Override
        public void wasRemovedApplication(ApplicationBundle applicationBundle) {
        }
    }

    public Activator() {
        this.addListener(new ListenerImpl());
    }

    @Override
    public void beforeStart(BundleContext bundleContext) {
        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                InteractiveShell.main();
            }
        });
        thread.start();
    }

    @Override
    public void startImpl(BundleContext bundleContext) {
    }

    @Override
    public void stopImpl(BundleContext bundleContext) {
    }
}
