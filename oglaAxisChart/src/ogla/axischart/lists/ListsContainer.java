package ogla.axischart.lists;

import java.util.ArrayList;
import java.util.List;

public class ListsContainer {

    private List<Object> objects = new ArrayList<Object>();

    public ListsContainer() {
    }

    private boolean isInContainer(Object obj) {
        for (Object object : objects) {
            if (obj == object) {
                return true;
            }
        }
        return false;
    }

    public void add(Object obj) {
        if (!isInContainer(obj)) {
            objects.add(obj);
        }
    }

    public void remove(Object obj) {
        objects.remove(obj);
    }

    public <T> T get(Class c) {
        for (Object object : objects) {
            T t = null;
            try {
                t = (T) c.cast(object);
            } catch (ClassCastException castException) {
                t = null;
            }
            if (t != null) {
                return t;
            }
        }
        return null;
    }

    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("[");
        for (int fa = 0; fa < objects.size(); fa++) {
            builder.append(objects.get(fa).toString());
            if (fa != objects.size() - 1) {
                builder.append(", ");
            }
        }
        builder.append("]");
        return builder.toString();
    }
}

