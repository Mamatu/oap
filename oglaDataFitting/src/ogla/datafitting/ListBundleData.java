
package ogla.datafitting;

import ogla.exdata.DataBundle;
import java.util.ArrayList;
import java.util.List;
import javax.swing.DefaultListModel;

/**
 *
 * @author marcin
 */
public class ListBundleData {

    public List<DataBundle> list = new ArrayList<DataBundle>();
    public DefaultListModel model = new DefaultListModel();

    private class Label {

        private DataBundle bundleData;

        public Label(DataBundle bundleData) {
            this.bundleData = bundleData;
        }

        @Override
        public String toString() {
            return this.bundleData.getLabel();
        }
    }

    public void add(DataBundle bundleData) {
        list.add(bundleData);
        model.addElement(new Label(bundleData));
    }

    public void remove(DataBundle bundleData) {
        int index = list.indexOf(bundleData);
        if (index != -1) {
            list.remove(index);
            model.remove(index);
        }
    }
}
