package ogla.axischart2D.plugin.chartcomponent;

import ogla.chart.DrawableOnChart;
import ogla.axischart2D.AxisChart2D;
import ogla.chart.DrawingInfo;
import java.awt.Graphics2D;
import javax.swing.JFrame;
import javax.swing.JPanel;

/**
 * Chart component which can be displayed on chart. 
 */
public abstract class AxisChart2DComponent implements DrawableOnChart {

    /**
     * Component is visible, or not
     */
    protected boolean isVisible = true;
    /**
     * BundleChartComponent to which this component is attached.
     */
    protected AxisChart2DComponentBundle bundleChartComponent = null;

    /**
     * Constructor whose param is BundleChartComponent to which this component is attached.
     * @param bundleChartComponent BundleChartComponent to which this component is attached
     */
    public AxisChart2DComponent(AxisChart2DComponentBundle bundleChartComponent) {
        this.bundleChartComponent = bundleChartComponent;
    }

    /**
     * Set visibility of this component
     * @param b
     */
    public void setVisible(boolean b) {
        isVisible = b;
    }

    /**
     * Attach chart to this component.
     * @param axisChart
     */
    public abstract void attachTo(AxisChart2D axisChart);

    /**
     * Save some information about this component to byte of array.
     * This method is used by AxisChart application to save information of component
     * to file before exit of application.
     * @return array of bytes in which is stored state of object.
     */
    public abstract byte[] save();

    /**
     * Used by AxisChart to display this component.
     * @param gd
     */
    public void drawOnChart(Graphics2D gd, DrawingInfo drawingType) {
        if (isVisible) {
            drawImpl(gd, drawingType);
        }
    }

    /**
     * Get frame which can be used to configuration this component.
     * @return JFrame or null
     */
    public abstract JFrame getFrame();

    /**
     * Get bundle to which is attached this component.
     * @return instance of BundleChartComponent
     */
    public AxisChart2DComponentBundle getBundle() {
        return bundleChartComponent;
    }

    /**
     * Method is used to drawn on chart surface.
     */
    public abstract void drawImpl(Graphics2D gd, DrawingInfo drawingType);
}
