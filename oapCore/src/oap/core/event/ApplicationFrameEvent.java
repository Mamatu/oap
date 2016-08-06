package ogla.core.event;

public interface ApplicationFrameEvent extends AnalysisEvent {

    /**
     * Event which is invoked during exiting of application.
     */
    public void isExited();

    /**
     * Event which is invoked after starting of application.
     */
    public void isStarted();
}
