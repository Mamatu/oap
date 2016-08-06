package ogla.axischart.plugin.plotmanager;

import ogla.core.ExtendedInformation;
import ogla.axischart.AxisChart;

public interface PlotManagerBundle extends ExtendedInformation {

    public String getSymbolicName();

    public PlotManager newPlotManager(AxisChart axisChart);
}
