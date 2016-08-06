package ogla.chart;

import java.awt.Graphics2D;

/**
 *  Class represent object which can be draw on chart's surface.
 */
public interface DrawableOnChartSurface {

    /**
     * Method which is used to draw.
     * @param gd
     * @param drawingInfo - type of drawing. @see DrawingInfo
     */
    public void drawOnChartSurface(Graphics2D gd, DrawingInfo drawingInfo);
}
