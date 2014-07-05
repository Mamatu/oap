package ogla.axischart.plugin.plotmanager;

import java.awt.Graphics2D;

public interface PlotManager {

    /**
     * Paint graphic symbol which identify this plot manager. This can be used during create legend of chart.
     * @param x
     * @param y
     * @param width
     * @param height
     * @param gd
     */
    public void getGraphicSymbol(int x, int y, int width, int height, Graphics2D gd);
}
