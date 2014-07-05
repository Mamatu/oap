package ogla.core.data;

import java.io.Serializable;
import javax.swing.JFrame;
import ogla.core.ExtendedInformation;
import ogla.core.EventSender;
import ogla.core.event.AnalysisEvent;

/**
 *  Class which is representation of bundle. It store all instances of RepositoryData which are
 * created by this bundle.
 */
public interface DataBundle extends ExtendedInformation, EventSender, Serializable {

    /**
     * Get current array of repositories data.
     * @return repositories data
     */
    public DataRepository get(int index);

    public int size();

    /**
     * Get if repositories of this bundle can be saved.
     * Return answer of this method influences on invoking of load method and save method.
     * @return true - if can, false - if can't
     */
    public boolean canBeSaved();

    /**
     * Load some data about this bundle which is saved in file.
     */
    public void loadBundleData(byte[] bytes);

    /**
     * Save some info of this bundle into file.
     * @return
     */
    public byte[] saveDataBundle();

    /**
     * Load byte and create new DataRepositoriesGroup object which is saved in container.
     * @param bytes byte representation 
     */
    public void loadToContainer(byte[] bytes);

    /**
     * Load byte and create new DataRepositoriesGroup object which is NOT saved in container.
     * @param bytes byte representation
     * @return RepositoryData
     */
    public DataRepository load(byte[] bytes);

    /**
     * Return unique symbolic name of this bundle.
     * @return
     */
    public String getSymbolicName();

    /**
     * Get label of this bundle.
     * @return label
     */
    public String getLabel();

    /**
     * Get descritption of this bundle.
     * @return descritption
     */
    public String getDescription();

    /**
     * Get gui of this bundle.
     * @return JFrame or null
     */
    public JFrame getFrame();

    public AnalysisEvent[] getEvents();
}
