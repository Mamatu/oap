
package ogla.datacsv.osgi;

import ogla.core.data.DataBundle;
import ogla.datacsv.DataFrame;
import org.osgi.framework.BundleActivator;
import org.osgi.framework.BundleContext;

/**
 *
 * @author Marcin
 */
public class Activator implements BundleActivator {

    public void start(BundleContext context) {

        DataFrame dataFrame=new DataFrame();

        DataBundle dataBundle = dataFrame.getDataBundle();
        context.registerService(DataBundle.class.getName(), dataBundle, null);
    }

    public void stop(BundleContext context) {
    }
}
