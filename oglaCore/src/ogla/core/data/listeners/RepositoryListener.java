package ogla.core.data.listeners;

import ogla.core.data.DataBlock;

/**
 * Listener which can be attachmented to some DataRepository object.
 */
public interface RepositoryListener {

    /**
     * Method should be invoked (it depends on vendor of DataRepository) after added this listener into some RepositoryData object.
     * It can be used to take some initialize informations from DataRepository object.
     * @param repository RepositoryData object into which this listener was added.
     */
    public void listenerIsAdded(DataBlock repository);

    /**
     * Method should be invoked (it depends on vendor of DataRepository) after remove this listener from RepositoryData object.
     * @param repository DataRepository object from which this listener was removed.
     */
    public void listenerIsRemoved(DataBlock repository);

    /**
     * Method should be invoked (it depends on vendor of DataRepository) after changed size of DataRepository container (when data was added or removed)
     * or/and if data is changed.
     * @param dataBlock DataRepository
     */
    public void isChangedSizeAndData(DataBlock dataBlock);

    /**
     * Method should be invoked (it depends on vendor of DataRepository) after changed
     * content of DataRepository container.
     * This method should be invoked only when content was changed without change of size this container.
     * @param dataBlock DataRepository
     */
    public void isChangedData(DataBlock dataBlock);
}
