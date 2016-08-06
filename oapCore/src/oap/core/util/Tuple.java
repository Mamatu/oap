
package ogla.core.util;

/**
 * Tuple which contains pair of varaibles (key -> variable).
 */
public class Tuple<T1, T2> {

    protected T1 object1;
    protected T2 object2;

    /**
     * Default constructor
     * @param object1 T1 object
     * @param object2 T2 object
     */
    public Tuple(T1 object1, T2 object2) {
        this.object1 = object1;
        this.object2 = object2;
    }

    /**
     *
     * @return T2 object
     */
    public T2 getObject2() {
        return object2;
    }

    /**
     *
     * @return T1 object
     */
    public T1 getObject1() {
        return object1;
    }

    /**
     * Return T1 object which is used as key.
     * @return T1 object
     */
    public T1 getKey() {
        return object1;
    }

    /**
     * Return T2 object which is used as variable.
     * @return T2 object
     */
    public T2 getValue() {
        return object2;
    }

    @Override
    public boolean equals(Object object) {
        if ((object instanceof Tuple)) {
            Tuple tuple = (Tuple) object;
            if (tuple.getObject2().equals(this.getObject2()) && tuple.getObject1().equals(this.getObject1())) {
                return true;
            }
        }
        return false;
    }
}
