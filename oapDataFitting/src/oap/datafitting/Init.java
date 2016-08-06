package ogla.datafitting;

import ogla.exdata.DataBundle;

/**
 *
 * @author marcin
 */
public class Init {

    protected DataFrame dataFrame = new DataFrame();

    public void add(DataBundle bundleData) {
        dataFrame.getListBundleData().add(bundleData);

    }

    public void remove(DataBundle bundleData) {
        dataFrame.getListBundleData().remove(bundleData);
    }

    public DataBundle getBundleData() {
        return dataFrame.getBundleData();
    }
}
