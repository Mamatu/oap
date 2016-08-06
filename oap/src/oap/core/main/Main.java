package ogla.core.main;

import java.io.File;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

class InitPlugins {

    private final String[] commands = {"--xargs props.xargs",
        "-Dorg.knopflerfish.gosg.jars=file:data/*", "-init"
    };
    private final String[] frameworkbundles = {
        "knopflerfish/log/log_all-2.0.2.jar",
        "knopflerfish/cm/cm_all-2.0.1.jar",
        "knopflerfish/component/component_all-2.0.0.jar",
        "knopflerfish/event/event_all-2.0.4.jar",
        "knopflerfish/prefs/prefs_all-2.0.3.jar",
        "knopflerfish/util/util-2.0.0.jar",
        "knopflerfish/crimson/crimson-2.0.1.jar",
        "knopflerfish/jsdk/jsdk_api-2.5.jar",
        "knopflerfish/remotefw/remotefw_api-2.0.0.jar"};
    final String[] cores = {
        "core/oglaCore-bundle.jar"
    };
    private String filePath = "";
    private String dataPath = "";

    public InitPlugins(String filePath, String dataPath) {
        this.filePath = filePath;
        this.dataPath = dataPath;
    }

    public String getContext() {
        String out = new String();
        for (String command : commands) {
            out += command + "\n";
        }
        File file = new File(dataPath);
        String[] bundles = file.list(new FilenameFilter() {
            public boolean accept(File dir, String name) {
                String end = ".jar";
                String fileend = String.valueOf(name.charAt(name.length() - 4));
                fileend += String.valueOf(name.charAt(name.length() - 3));
                fileend += String.valueOf(name.charAt(name.length() - 2));
                fileend += String.valueOf(name.charAt(name.length() - 1));
                if (fileend.contentEquals(end)) {
                    return true;
                }
                return false;
            }
        });
        for (String bundle : cores) {
            out += "-install " + this.dataPath + "/" + bundle + "\n";
        }
        for (String bundle : cores) {
            out += "-start " + this.dataPath + "/" + bundle + "\n";
        }
        if (bundles != null) {
            for (String bundle : bundles) {
                out += "-install " + this.dataPath + "/plugins/" + bundle + "\n";
            }
            for (String bundle : bundles) {
                out += "-start " + this.dataPath + "/plugins/" + bundle + "\n";
            }
        }

        return out;
    }

    public void createFile() {
        File file = new File(filePath);
        file.delete();
        try {
            file.createNewFile();
        } catch (IOException ioe) {
            System.err.println(ioe.getMessage());
        }
        FileWriter writer = null;
        try {
            writer = new FileWriter(filePath);
        } catch (IOException ioe) {
            System.err.println(ioe.getMessage());
        }
        String in = getContext();
        if (writer != null && in != null && in.length() > 0) {
            try {
                writer.write(in, 0, in.length());
                writer.close();
            } catch (IOException ioe) {
                System.err.println(ioe.getMessage());
            }
        }
    }
}

public class Main {

    static class ExceptionHandler implements Thread.UncaughtExceptionHandler {

        @Override
        public void uncaughtException(Thread t, Throwable e) {
            handle(e);
        }

        public void handle(Throwable throwable) {
            throwable.printStackTrace();
        }

        public static void registerExceptionHandler() {
            Thread.setDefaultUncaughtExceptionHandler(new ExceptionHandler());
            System.setProperty("sun.awt.exception.handler", ExceptionHandler.class.getName());
        }
    }

    public static void main(final String[] args) {
        try {
            new InitPlugins("osgi/init.xargs", "../data").createFile();
            Thread frameworkThread = new Thread(new Runnable() {
                @Override
                public void run() {
                    String[] params = new String[]{"-xargs", "osgi/init.xargs"};
                    org.knopflerfish.framework.Main.main(params);
                }
            });
            ExceptionHandler.registerExceptionHandler();
            frameworkThread.start();
        } catch (Exception ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
