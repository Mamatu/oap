/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.groovy;

import groovy.lang.Binding;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.codehaus.groovy.tools.shell.IO;

/**
 *
 * @author mmatula
 */
public class InteractiveShell {

    private static org.codehaus.groovy.tools.shell.Groovysh groovysh = null;

    static public void create(final String[] args) {
        try {
            Binding binding = new Binding();
            ObjectsManager.init(binding);
            groovysh = new org.codehaus.groovy.tools.shell.Groovysh(binding, new IO());
            boolean portUserDefined = true;
            int port = 4456;
            new ObjectsManager.RpcObjectImpl(binding);
            if (args.length > 0) {
                try {
                    port = Integer.parseInt(args[0]);
                } catch (NumberFormatException nfe) {
                    portUserDefined = false;
                }
            }
            ObjectsManager.getRpc().init(port);
            class GroovyRunnable implements Runnable {

                boolean portUserDefined = false;
                String[] args = null;

                public GroovyRunnable(boolean portUserDefined, String[] args) {
                    this.portUserDefined = portUserDefined;
                    this.args = args;
                }

                @Override
                public void run() {
                    if (portUserDefined == true) {
                        if (this.args.length > 1) {
                            String[] args1 = new String[this.args.length - 1];
                            System.arraycopy(this.args, 1, args1, 0, this.args.length - 1);
                            this.args = args1;
                        } else {
                            this.args = null;
                        }
                    }
                    groovysh.run(this.args);
                }
            }
            Thread groovyThread = new Thread(new GroovyRunnable(portUserDefined, args));
            groovyThread.start();
        } catch (IOException ex) {
            Logger.getLogger(InteractiveShell.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static void main() {
        String[] args = {""};
        InteractiveShell.create(args);
    }
}
