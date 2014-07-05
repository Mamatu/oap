package ogla.core.data.utils;

import java.util.ArrayList;
import java.util.List;
import ogla.core.data.DataRepository;
import ogla.core.data.History;

/**
 *
 * Simple default implementation of history.
 */
public class DefaultHistoryImpl extends History {

    private DataRepository currentGroup;

    public DefaultHistoryImpl(DataRepository currentGroup) {
        this.currentGroup = currentGroup;
    }
    private List<DataRepository> list = new ArrayList<DataRepository>();

    /**
     * Add DataRepository object into container.
     * @param dataRepositoryGroup
     */
    public void add(DataRepository dataRepositoryGroup) {
        list.add(dataRepositoryGroup);
    }

    /**
     * Push DataRepository object into container.
     * @param dataRepositoryGroup
     */
    public void push(DataRepository dataRepositoryGroup) {
        list.add(list.size(), dataRepositoryGroup);
    }

    /**
     * Get size of container.
     * @return
     */
    public int size() {
        return list.size();
    }

    /**
     * Get repositoryData from index
     * @param index
     * @return
     */
    public DataRepository get(int index) {
        return list.get(index);
    }

    @Override
    public DataRepository getCurrent() {
        return currentGroup;
    }
}
