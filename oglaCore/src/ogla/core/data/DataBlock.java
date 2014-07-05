package ogla.core.data;

import ogla.core.data.listeners.RepositoryListener;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Class which represent set of rows. Instances of this class split repository in
 * some number of blocks, which can be taken into account durin painting/plotting process.
 * 
 */
public interface DataBlock extends Serializable {

    /**
     * Get row of data at index.
     * @param index index in container
     * @return RowData object
     */
    public DataValue get(int row, int column);

    /**
     * Get height of container which contains columns.
     * @return
     */
    public int rows();

    public int columns();

    /**
     * Get repository to which belong this block.
     * @return parent repository
     */
    public DataRepository getDataRepository();
}
