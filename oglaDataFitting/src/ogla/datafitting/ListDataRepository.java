
package ogla.datafitting;

import ogla.exdata.DataRepository;
import ogla.exdata.DataBlock;
import java.util.ArrayList;
import java.util.List;
import javax.swing.DefaultListModel;

public class ListDataRepository {

    public List<DataRepository> list = new ArrayList<DataRepository>();
    public DefaultListModel model = new DefaultListModel();

    private class Label {

        private DataRepository dataRepositoriesGroup;

        public Label(DataRepository dataRepositoriesGroup) {
            this.dataRepositoriesGroup = dataRepositoriesGroup;
        }

        @Override
        public String toString() {
            return this.dataRepositoriesGroup.getLabel();
        }
    }

    public void setArray(DataRepository[] dataRepositoriesGroups) {
        list.removeAll(list);
        model.removeAllElements();
        for (DataRepository repositoryData : dataRepositoriesGroups) {
            this.add(repositoryData);
        }
    }

    public void add(DataRepository dataRepositoriesGroup) {
        list.add(dataRepositoriesGroup);
        model.addElement(new Label(dataRepositoriesGroup));
    }

    public void remove(DataRepository dataRepositoriesGroup) {
        int index = list.indexOf(dataRepositoriesGroup);
        if (index != -1) {
            list.remove(index);
            model.remove(index);
        }
    }

    public void remove(int index) {
        list.remove(index);
        model.remove(index);
    }
}

