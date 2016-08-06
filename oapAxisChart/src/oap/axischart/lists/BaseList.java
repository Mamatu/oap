package ogla.axischart.lists;

import java.util.ArrayList;
import java.util.List;

public class BaseList<T> implements ListInterface<T> {

    protected List<ListListener> listeners = new ArrayList<ListListener>();

    public synchronized void addListListener(ListListener listListener) {
        listeners.add(listListener);
        for (T t : list) {
            listListener.isAdded(t);
        }
    }

    public synchronized void removeListListener(ListListener listListener) {
        listeners.remove(listListener);
    }
    protected List<T> list = new ArrayList<T>();

    public BaseList() {
    }

    private boolean onlyOneInstance = false;

    public BaseList(boolean onlyOneInstance) {
        this.onlyOneInstance = onlyOneInstance;
    }

    private boolean isInstance(T t) {
        for (T comp : list) {
            if (comp == t) {
                return true;
            }
        }
        return false;
    }

    public synchronized boolean add(T t) {
        if (onlyOneInstance) {
            if (isInstance(t)) {
                return false;
            }
        }
        boolean b = list.add(t);
        if (b) {
            for (ListListener listener : listeners) {
                listener.isAdded(t);
            }
        }
        return b;
    }

    public synchronized int remove(T t) {
        int i = list.indexOf(t);
        if (i != -1) {
            list.remove(i);
            for (ListListener listener : listeners) {
                listener.isRemoved(t);
            }
            return i;
        }
        return -1;
    }

    public synchronized T remove(int index) {
        T t = list.remove(index);
        if (t != null) {
            for (ListListener listener : listeners) {
                listener.isRemoved(t);
            }
            return t;
        }
        return null;
    }

    public synchronized int size() {
        return list.size();
    }

    public synchronized T get(int index) {
        return list.get(index);
    }
}
