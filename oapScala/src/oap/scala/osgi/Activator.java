package ogla.scala.osgi;

import org.osgi.framework.BundleActivator;
import org.osgi.framework.BundleContext;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author mmatula
 */
public class Activator implements BundleActivator {

    private scala.Console console = null;

    @Override
    public void start(BundleContext bc) throws Exception {
        new scala.tools.nsc.interpreter.IMain();
    }

    @Override
    public void stop(BundleContext bc) throws Exception {

    }

}
