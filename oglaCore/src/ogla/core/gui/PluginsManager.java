package ogla.core.gui;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.osgi.framework.BundleContext;
import org.osgi.framework.BundleException;

/**
 *
 * @author marcin
 */
public class PluginsManager {

    BundleContext bundleContext;

    public PluginsManager(BundleContext bundleContext) {
        this.bundleContext = bundleContext;
    }

    Set<Plugin> startedPlugins = new LinkedHashSet<Plugin>();
    Set<Plugin> installedPlugins = new LinkedHashSet<Plugin>();
    private Set<Plugin> invalidPlugins = new LinkedHashSet<Plugin>();
    private StringBuilder error = new StringBuilder();

    public void removeInvalidPlugins() {
        for (Plugin plugin : invalidPlugins) {
            this.installedPlugins.remove(plugin);
        }
    }

    Set<Plugin> getInvalidPlugins(Set<Plugin> out) {
        out.clear();
        out.addAll(installedPlugins);
        return out;
    }

    void start() {
        invalidPlugins.clear();
        this.start(installedPlugins);
    }

    private List<BundleException> bundleExceptions = new ArrayList<BundleException>();

    private void start(Set<Plugin> plugins) {
        bundleExceptions.clear();
        Set<Plugin> plugins1 = new LinkedHashSet<Plugin>();
        for (Plugin plugin : plugins) {
            if (plugin.isStarted() == false) {
                try {
                    plugin.start();
                } catch (BundleException bex) {
                    plugins1.add(plugin);
                    bundleExceptions.add(bex);
                }
            }
        }
        if (plugins1.size() > 0 && plugins1.size() != plugins.size()) {
            this.start(plugins1);
        } else if (plugins1.size() == plugins.size()) {
            for (Plugin plugin : plugins) {
                this.appendErrorMsg("Error during starting: " + plugin.getPath() + ". You should contact to provider of the plugin.");
                invalidPlugins.add(plugin);
            }
            for (BundleException bex : bundleExceptions) {
                Logger.getLogger(PluginsManager.class.getName()).log(Level.SEVERE, null, bex);
            }
        } else {

        }
    }

    void appendErrorMsg(String l) {
        error.append("--- ");
        error.append(l);
        error.append("\n");
    }

    public String getErrorMsg() {
        if (error.length() == 0) {
            return "";
        }
        String o = "Errors:\n" + error.toString();
        error.delete(0, error.capacity());
        return o;
    }
}
