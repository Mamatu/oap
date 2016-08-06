
package ogla.core.util;

/**
 * Simple class with personal identificator.
 * @author marcin
 * @param <T>
 */
public interface GenericArray<T> {

    public T get(int index);

    public int size();

    public String id();
}
