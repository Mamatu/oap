package ogla.datacsv;

import ogla.core.data.DataBlock;
import ogla.core.data.DataRepository;
import ogla.core.data.DataValue;
import ogla.core.data.listeners.RepositoryListener;
import ogla.core.data.utils.DataTools.DataRowImpl;
import java.util.ArrayList;
import java.util.List;

public class DataBlockImpl implements DataBlock {

    public List<DataRowImpl> rows = new ArrayList<DataRowImpl>();
    public List<RepositoryListener> listeners = new ArrayList<RepositoryListener>();

    public DataBlockImpl(DataRepository dataRepositoryImpl) {
        this.dataRepository = dataRepositoryImpl;
    }

    public DataValue get(int row, int column) {
        return rows.get(row).get(column);
    }

    public int rows() {
        return rows.size();

    }

    public int columns() {
        int w = 0;
        for (DataRowImpl row : rows) {
            if (w < row.size()) {
                w = row.size();
            }
        }
        return w;

    }

    public void addRepositoryListener(RepositoryListener listener) {
        listeners.add(listener);
    }

    public void removeRepositoryListener(RepositoryListener listener) {
        listeners.remove(listener);
    }
    public DataRepository dataRepository;

    public DataRepository getDataRepository() {
        return dataRepository;
    }
}
