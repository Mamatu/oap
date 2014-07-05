package ogla.axischart.util;

import ogla.core.util.ListElementManager;
import ogla.core.BasicInformation;
import java.awt.Color;
import java.awt.Component;
import javax.swing.JCheckBox;
import javax.swing.JList;
import javax.swing.ListCellRenderer;

public class CheckList extends JCheckBox implements ListCellRenderer {

    public Component getListCellRendererComponent(JList list, Object value, int index, boolean isSelected, boolean cellHasFocus) {
        if (!(value instanceof Element)) {
            return null;
        }
        this.setText(value.toString());
        if (isSelected) {
            this.setBackground(Color.lightGray);
        } else {
            this.setBackground(Color.white);
        }
        this.setSelected(((Element) value).isAvtive);
        return this;
    }

    public static class Element extends ListElementManager.Element {

        public boolean isAvtive = true;

        public Element(Object object, String label) {
            super(object, label);
        }

        public Element(BasicInformation basicInformation) {
            super(basicInformation);
        }

        public Object getObject() {
            if (object instanceof ListElementManager.Element) {
                ListElementManager.Element listElement = (ListElementManager.Element) object;
                return listElement.object;
            } else {
                return object;
            }
        }

        @Override
        public String toString() {
            return label;
        }
    }
}
