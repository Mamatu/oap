package ogla.core.application;

import ogla.core.EventSender;
import ogla.core.ExtendedInformation;

/**
 * Class whose objects are registered in OSGi ServiceRegister and play role of
 * applications.
 */
public interface ApplicationBundle extends ExtendedInformation, EventSender {

    /**
     * Create new instance of Application.
     *
     * @return new instance of Application
     */
    public abstract Application newApplication();

    /**
     * Get if application of this bundle can be saved. Return answer of this
     * method influences on invoking of load method and save method.
     *
     * @return true - if can, false - if can't
     */
    public abstract boolean canBeSaved();

    /**
     * Load application from bytes array. Array of bytes comes from previous
     * saved application. Goal of this method is created Application object
     * which contains state previous saved.
     *
     * @param bytes bytes representation of application. Content of this is
     * equal to array which was returned by
     * {@link ogla.exapplication.application.Application#save()}.
     * @return instance of Application
     * @see ogla.exapplication.application.Application#save()
     */
    public abstract Application load(byte[] bytes);

    /**
     * Get label of this bundle.
     *
     * @return label
     */
    @Override
    public abstract String getLabel();

    /**
     * Get unique symbolic name for this bundle.
     *
     * @return symbolic name
     */
    public abstract String getSymbolicName();

    /**
     * Get description of this bundle.
     *
     * @return description
     */
    @Override
    public abstract String getDescription();
}
