package ogla.core.gui;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Dictionary;
import java.util.List;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.tree.DefaultMutableTreeNode;
import org.osgi.framework.Bundle;
import org.osgi.framework.BundleContext;
import org.osgi.framework.BundleException;
import sun.awt.UNIXToolkit;

public class Plugin implements Comparable<Plugin> {

    private final String PluginSection = "Plugin-Section";
    private final String ExportPackages = "Export-Package";
    private final String ImportPackages = "Import-Package";
    private final String SymbolicName = "Bundle-SymbolicName";
    private final String Name = "Bundle-Name";
    private final String Version = "Bundle-Version";
    private final String Author = "Bundle-Vendor";
    private final String Contact = "Plugin-Contact";
    private final String State = "Plugin-State";
    private final String Webhome = "Plugin-Webhome";
    private DefaultMutableTreeNode treeNode = new DefaultMutableTreeNode(this);
    private String name = "";
    private String description = "";
    private String section = "";
    private String path = "";
    private String version = "";
    private String symbolicName = "";
    private String author = "";
    private String state = "";
    private String contact = "";
    private String webHome = "";
    private Bundle bundle = null;
    private String[] exportPackages = null;
    private String[] importPackages = null;
    private boolean installed = false;
    private boolean started = false;
    private PluginsManager pluginsManager = null;
    private List<Listener> listeners = new ArrayList<Listener>();
    private Lock listenerLock = new ReentrantLock();

    public interface Listener {

        public enum ActionType {

            UNINSTALL,
            INSTALL,
            STOP,
            START
        }

        public void onAction(Plugin plugin, ActionType actionType);
    }

    public void addListener(Listener listener) {
        listenerLock.lock();
        this.listeners.add(listener);
        listenerLock.unlock();
    }

    public void removeListener(Listener listener) {
        listenerLock.lock();
        this.listeners.remove(listener);
        listenerLock.unlock();
    }

    private void setState(Listener.ActionType actionType) {
        switch (actionType) {
            case UNINSTALL:
                this.state = "uninstalled";
                break;
            case INSTALL:
                this.state = "installed";
                break;
            case START:
                this.state = "started";
                break;
            case STOP:
                this.state = "stoped";
                break;
        }
    }

    private void executeListener(Listener.ActionType actionType) {
        this.setState(actionType);
        listenerLock.lock();
        for (Listener listener : listeners) {
            listener.onAction(this, actionType);
        }
        listenerLock.unlock();
    }

    Plugin(PluginsManager pluginsManager) {
        this.pluginsManager = pluginsManager;
    }

    String[] getExportPackages() {
        return this.exportPackages;
    }

    String[] getImportPackages() {
        return this.importPackages;
    }

    String[] getDependecies() {
        return null;
    }

    String getPath() {
        return path;
    }

    DefaultMutableTreeNode getTreeNode() {
        return treeNode;
    }

    public boolean isInstalled() {
        return this.installed;
    }

    public boolean isStarted() {
        return this.started;
    }

    public String getName() {
        return this.name;
    }

    public String getSymbolicName() {
        return this.symbolicName;
    }

    public String getSection() {
        return this.section;
    }

    public String getDescription() {
        return this.description;
    }

    public String getVersion() {
        return this.version;
    }

    public String getState() {
        return this.state;
    }

    public String getContact() {
        return this.contact;
    }

    public String getAuthor() {
        return this.author;
    }

    public String getWebHome() {
        return this.webHome;
    }

    private String getValue(String name, Dictionary<String, String> dict) {
        String value = (String) dict.get(name);
        if (value == null) {
            value = "";
        }
        return value;
    }

    private boolean checkPathes(String path) {
        boolean output = true;
        for (Plugin fa : pluginsManager.installedPlugins) {
            if (path.equals(fa.path)) {
                output = false;
                break;
            }
        }
        return output;
    }

    private boolean checkPlugin(Plugin plugin) {
        boolean output = true;
        if (plugin.installed == false) {
            return false;
        }
        for (Plugin fa : pluginsManager.installedPlugins) {
            if (fa != plugin) {
                if (plugin.getSymbolicName().length() > 0 && plugin.version.length() > 0
                        && plugin.getSymbolicName().equals(fa.getSymbolicName()) && plugin.version.equals(fa.version)) {
                    output = false;
                    break;
                }
            }
        }
        return output;
    }

    private String[] getValues(String key, Dictionary<String, String> dict) {
        String value = (String) dict.get(key);
        String[] array = null;
        if (value != null && value.equals("") == false) {
            String[] paths = value.split(",");
            array = paths;
        }
        return array;
    }

    public int indexOfExportPackage(String packagePath) {
        int index = -1;
        for (int fa = 0; fa < exportPackages.length; fa++) {
            if (exportPackages[fa].equals(packagePath) == true) {
                index = fa;
                break;
            }
        }
        return index;
    }

    public int indexOfImportPackage(String packagePath) {
        int index = -1;
        for (int fa = 0; fa < importPackages.length; fa++) {
            if (importPackages[fa].equals(packagePath) == true) {
                index = fa;
                break;
            }
        }
        return index;
    }

    public Plugin getInstalledPlugin(String packagePath) {
        for (Plugin plugin : pluginsManager.installedPlugins) {
            if (plugin.indexOfExportPackage(packagePath) > -1) {
                return plugin;
            }
        }
        return null;
    }

    public Plugin getStartedPlugin(String packagePath) {
        for (Plugin plugin : pluginsManager.startedPlugins) {
            if (plugin.indexOfExportPackage(packagePath) > -1) {
                return plugin;
            }
        }
        return null;
    }

    public class DependenceIsNotFoundException extends Exception {

        public DependenceIsNotFoundException(String title) {
            super(title + " was not installed.");
        }
    }

    public boolean install(String path, BundleContext bundleContext) {
        boolean status = false;
        if (checkPathes(path) == true) {
            FileInputStream fileInputStream = null;
            try {
                fileInputStream = new FileInputStream(new File(path));
                Bundle bundle = bundleContext.installBundle(path, fileInputStream);
                bundle.getServicesInUse();
                if (bundle != null) {
                    this.installed = true;
                    Dictionary<String, String> dict = bundle.getHeaders();
                    this.exportPackages = getValues(ExportPackages, dict);
                    this.importPackages = getValues(ImportPackages, dict);
                    this.bundle = bundle;
                    this.section = getValue(PluginSection, dict);
                    this.name = getValue(Name, dict);
                    this.version = getValue(Version, dict);
                    this.author = getValue(Author, dict);
                    this.contact = getValue(Contact, dict);
                    this.webHome = getValue(Webhome, dict);
                    this.state = getValue(State, dict);
                    this.symbolicName = getValue(SymbolicName, dict);
                    this.path = path;
                    if (this.name.length() == 0) {
                        this.name = this.path.substring(this.path.lastIndexOf('/') + 1, this.path.length());
                    }
                    if (checkPlugin(this) == true) {
                        pluginsManager.installedPlugins.add(this);
                        executeListener(Listener.ActionType.INSTALL);
                        status = true;
                    } else {
                        this.uninstall(false);
                    }
                }
            } catch (FileNotFoundException ex) {
                Logger.getLogger(Plugin.class.getName()).log(Level.SEVERE, null, ex);
                pluginsManager.appendErrorMsg("File not found: " + ex.getMessage());
            } catch (BundleException ex) {
                Logger.getLogger(Plugin.class.getName()).log(Level.SEVERE, null, ex);
                pluginsManager.appendErrorMsg("Bundle exception: " + ex.getMessage());
            } finally {
                try {
                    fileInputStream.close();
                } catch (IOException ex) {
                    Logger.getLogger(Plugin.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        }
        return status;
    }

    public boolean uninstall() throws BundleException {
        return uninstall(true);
    }

    private boolean uninstall(boolean executeListener) throws BundleException {
        boolean status = false;
        if (this.bundle != null && this.installed) {
            this.bundle.uninstall();
            if (executeListener) {
                executeListener(Listener.ActionType.UNINSTALL);
            }
            this.installed = false;
            status = true;
            pluginsManager.installedPlugins.remove(this);
        }
        return status;
    }

    public boolean start() throws BundleException {
        boolean status = true;
        try {
            this.bundle.start();
            pluginsManager.startedPlugins.add(this);
            executeListener(Listener.ActionType.START);
        } catch (BundleException bundleException) {
            status = false;
            throw bundleException;
        } finally {
            this.started = status;
        }
        return status;
    }

    public boolean stop() {
        boolean status = true;
        try {
            this.bundle.stop();
            executeListener(Listener.ActionType.STOP);
        } catch (BundleException ex) {
            Logger.getLogger(Plugin.class.getName()).log(Level.SEVERE, null, ex);
            status = false;
        }
        return status;
    }

    public int compareTo(Plugin o) {
        if (path.equals(o.path)) {
            return 0;
        }
        return 1;
    }

    @Override
    public String toString() {
        return name;
    }
}
