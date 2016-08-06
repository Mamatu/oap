package ogla.core.data;

import ogla.core.data.listeners.RepositoryListener;
import java.io.Serializable;
import ogla.core.BasicInformation;

/**
 * Repository of data which stores some number of DataBlocks.
 * @author marcin
 */
public interface DataRepository extends BasicInformation, Serializable {

    public DataBlock get(int index);

    public int size();

    /**
     * Add repository listener into container of listeners.
     * @param listener
     */
    public void addRepositoryListener(RepositoryListener listener);

    /**
     * Remove repository listener from container of listeners.
     * @param listener
     */
    public void removeRepositoryListener(RepositoryListener listener);

    /**
     * Get bundle data to which belong this repository.
     * @return
     */
    public abstract DataBundle getBundle();

    /**
     * Get history of this repository group. History's object contains previous states of this group.
     * @return history or null if this group doesn't supported history
     */
    public abstract History getHistory();

    /**
     * Save state of this repository in array of bytes.
     * This method is used by core to save content of repository
     * to file before exit of application. Saved are all reposiories which was created
     * during work of application.
     * @return array of bytes in which is stored state of object.
     */
    public abstract byte[] save();

    /**
     * Get label of this repository.
     * @return label of repository
     */
    public abstract String getLabel();

    /**
     * Get description of this repository.
     * @return description of repository
     */
    public abstract String getDescription();
}
