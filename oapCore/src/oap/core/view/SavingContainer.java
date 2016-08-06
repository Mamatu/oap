package ogla.core.view;

import ogla.core.application.Application;
import ogla.core.application.ApplicationBundle;

/**
 * Class which should be used to comunication beetwen AnalysisCore and BundleView.
 * Instance of this class has method which can be used to send information about saved application.
 * 
 * @author marcin
 */
public interface SavingContainer {

    /**
     * Save application.
     * @see ogla.exview.osgi.BundleView#saveApplication(SavingContainer saveContainer)
     * @param application saved application
     * @param bundleApplication bundle to which belong saved application
     * @param label label/name of your application which will be used as text displayed in window of loading.
     *
     */
    public void save(Application application, ApplicationBundle bundleApplication, String label);
}
