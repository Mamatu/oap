
package ogla.axischart2D.plugin.chartcomponent;

import ogla.core.ExtendedInformation;

/**
 * 
 */
public interface AxisChart2DComponentBundle extends ExtendedInformation {

    /**
     * Create new instance of {@link ogla.axischart.components.chart.ChartComponent} .
     * @return
     */
    public AxisChart2DComponent newAxisChart2DComponent();

    /**
     * Create instance of this class by used information which are
     * contained in byte[] array.
     * @return
     */
    public AxisChart2DComponent load(byte[] bytes);

    /**
     * ChartComponents of this class can be saved or no. If true, core of application try use
     * {@link ogla.axischart.components.chart.ChartComponent#save()} to saved instances of 
     * ChartComponent.
     * @see ogla.axischart.components.chart.ChartComponent#save()
     * @return true - if can be saved, false - it can't be saved
     */
    public boolean canBeSaved();

    /**
     * Get unique symbolic name for this bundle.
     * @return symbolic name
     */
    public String getSymbolicName();
}
