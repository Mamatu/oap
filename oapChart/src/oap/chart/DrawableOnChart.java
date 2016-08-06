package ogla.chart;


import java.awt.Graphics2D;
import ogla.core.BasicInformation;

/**
 *  Class represent object which can be draw on chart.
 * 
 */
public interface DrawableOnChart extends BasicInformation {

    /**
     * Method which is used to draw on chart.
     * @param gd
     * @param drawingInfo - type of drawing. @see DrawingInfo
     */
    public void drawOnChart(Graphics2D gd, DrawingInfo drawingInfo);
}
