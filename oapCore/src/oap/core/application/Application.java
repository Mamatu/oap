package ogla.core.application;

import javax.swing.JPanel;
import ogla.core.ui.ObjectsList;
import ogla.core.ui.OglaObject;

/**
 *
 * @author marcin
 */
public interface Application {

    OglaObject getOglaObject(int index);

    int getOglaObjectsCount();

    /**
     * Get instance BundleApplication of this application.
     *
     * @return
     */
    ApplicationBundle getBundle();

    /**
     * Get JPanel which contains workspace of application. Can be null.
     *
     * @return
     */
    public JPanel getAppPanel();

    /**
     * This method is used by core to save array of bytes (some information
     * about application) to file before exit of application. All instances of
     * applications are saved which was marked (saved) by user.
     *
     * @return array of bytes in which is stored state of object.
     */
    public byte[] save();
}
