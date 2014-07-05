
package ogla.axischart.plugin.image;

import ogla.core.ExtendedInformation;


/**
 * Bundle which should be registered in OSGi.
 */
public interface ImageComponentBundle extends ExtendedInformation {

    /**
     * Create new instance of this class.
     * @return
     */
    public ImageComponent newImageComponent();

    /**
     * Create instance of this class by used information which are
     * contained in byte[] array.
     * @return
     */
    public ImageComponent load(byte[] bytes);

    /**
     * ImageComponents of this class can be saved or no.
     * @return
     */
    public boolean canBeSaved();

    /**
     * Get unique symbolic name for this bundle.
     * @return symbolic name
     */
    public String getSymbolicName();
}
