
package ogla.core.view;


import java.awt.Component;
import javax.swing.JPanel;
import ogla.core.BasicInformation;
import ogla.core.EventSender;
import ogla.core.application.Application;
import ogla.core.event.AnalysisEvent;

/**
 * Interface which is used to create customize view of data content.
 * 
 */
public interface ViewBundle extends BasicInformation, EventSender {

    /**
     * Get main panel.
     * @return panel which is displayed.
     */
    public JPanel getMainView();

    /**
     * Save application.
     * @param savingContainer 
     */
    public void saveApplication(SavingContainer savingContainer);

    /**
     * Callback method. Is invoked during loaded application from loading window.
     * @param application
     * @param label
     */
    public void isLoadedApplication(Application application, String label);

    /**
     * 
     * @return Components which should be update when LookAndFeel is changed.
     */
    public Component[] getComponentsToUpdateUI();

    /**
     * Get label of this view.
     * @return label
     */
    public String getLabel();

    /**
     * Get description of this view
     * @return description
     */
    public String getDescription();

    public AnalysisEvent[] getEvents();
}
