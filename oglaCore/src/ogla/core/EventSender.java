
package ogla.core;

import ogla.core.event.AnalysisEvent;

public interface EventSender {

    /**
     * Get events which are necessary for this bundle.
     * @see ogla.excore.event.AnalysisEvent
     * @return array of AnalysisEvents or null
     */
    public AnalysisEvent[] getEvents();
}
