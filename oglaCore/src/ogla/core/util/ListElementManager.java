package ogla.core.util;

import ogla.core.BasicInformation;
import javax.swing.ListModel;

public class ListElementManager {

    public static class Element {

        public Object object;
        public String label = "";

        public Element(BasicInformation basicInformation) {
            this.object = basicInformation;
            this.label = basicInformation.getLabel();
        }

        public Element(Object object, String label) {
            this.object = object;
            this.label = label;
        }

        @Override
        public String toString() {
            return label;
        }
    }

    public static Element register(BasicInformation basicInformation) {
        Element rawElement = new Element(basicInformation);
        return rawElement;
    }

    public static Element register(Object object, String label) {
        Element rawElement = new Element(object, label);
        return rawElement;
    }

    public static int getIndex(Object object, ListModel listModel) {
        for (int fa = 0; fa < listModel.getSize(); fa++) {
            Object obj = listModel.getElementAt(fa);
            if (obj instanceof Element) {
                Element element = (Element) obj;
                if (element.object.equals(object)) {
                    return fa;
                }
            }
            if (object.equals(obj)) {
                return fa;
            }
        }
        return -1;
    }

    public static Object get(Object object) {
        if (!(object instanceof Element)) {
            return object;
        }
        Element rawElement = (Element) object;
        return rawElement.object;

    }
}
