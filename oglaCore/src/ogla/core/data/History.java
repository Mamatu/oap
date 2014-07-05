package ogla.core.data;

import java.io.Serializable;

/**
 * Object of this class contains repositories which represent previous states of repository to which
 * object is attached.
 * @author marcin
 */
public abstract class History implements Serializable {

    /**
     *
     * @return size of history
     */
    public abstract int size();

    /**
     * Get previous repository. The first DataRepository object should be the youngest.
     * @param index
     * @return
     */
    public abstract DataRepository get(int index);

    public abstract DataRepository getCurrent();

    public DataRepository getTheOldest() {
        i = size() - 1;
        return get(size() - 1);
    }

    public DataRepository getTheYoungest() {
        i = 0;
        return get(0);
    }
    private int i = -1;

    public DataRepository get() {
        if (i == -1) {
            return getCurrent();
        }
        return get(i);
    }

    public boolean isCurrent() {
        if (i == -1) {
            return true;
        }
        return false;
    }

    public void current() {
        i = -1;
    }

    public DataRepository older() {
        if (i == size() - 1) {
            return null;
        }
        i++;
        return get(i);
    }

    public DataRepository younger() {
        i--;
        if (i == -1) {
            i = -1;
            return getCurrent();
        } else if (i < -1) {
            return null;
        }
        return get(i);
    }
}
