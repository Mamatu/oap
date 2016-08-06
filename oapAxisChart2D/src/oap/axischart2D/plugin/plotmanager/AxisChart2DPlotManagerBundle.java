package ogla.axischart2D.plugin.plotmanager;

import ogla.core.ExtendedInformation;
import ogla.axischart.AxisChart;
import ogla.axischart.plugin.plotmanager.PlotManagerBundle;
import ogla.axischart2D.AxisChart2D;

/**
 * Class which should be registered in OSGi.
 */
public abstract class AxisChart2DPlotManagerBundle implements PlotManagerBundle {

    /**
     * Get unique symbolic name for this bundle.
     * @return symbolic name
     */
    public abstract String getSymbolicName();

    /**
     * PlotManagers of this bundle can be saved or no. If true, AxisChart try use
     * {@link ogla.axischart.components.plotmanager.PlotManager#save()} to saved instances of
     * ChartComponent.
     * @see ogla.axischart.components.plotmanager.PlotManager#save()
     * @return true - if can be saved, false - it can't be saved
     */
    public abstract boolean canBeSaved();

    /**
     * Get new instance of PlotManager whose stome information was saved.
     * @param axisChart
     * @param bytes array which contains information about PlotManager
     * @return new instance of PlotManager which store information saved in bytes.
     */
    public abstract AxisChart2DPlotManager load(AxisChart2D axisChart, byte[] bytes);

    /**
     * Get new instance of PlotManager.
     * @param axisChart - chart to which will be attached PlotManager.
     * @return new instance of PlotManager
     */
    public abstract AxisChart2DPlotManager newPlotManager(AxisChart axisChart);

    /**
     * Get label of this bundle.
     * @return
     */
    public abstract String getLabel();

    @Override
    public String toString() {
        return this.getLabel();
    }
}
