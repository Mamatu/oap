
package ogla.axis.util;

import ogla.core.data.DataValue;
import ogla.axischart2D.AxisChart2D;
import ogla.axischart2D.Ticks;
import ogla.axischart2D.coordinates.CoordinateTransformer;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Stroke;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.awt.geom.Line2D;


public abstract class CursorLocation {

    protected LocationManager locationManager;
    protected DataValue axisTick = null;
    protected DataValue oldAxisTick = null;
    protected Line2D.Double cursorLine = new Line2D.Double();
    protected String label = "";
    protected AxisChart2D axisChart;
    protected CoordinateTransformer transformer;
    protected MouseMotionListenerImpl mouseMotionListenerImpl;

    public CursorLocation(AxisChart2D axisChart, Ticks ticks, CoordinateTransformer transformer) {
        this.axisChart = axisChart;
        this.transformer = transformer;
        this.locationManager = new LocationManager(axisChart, ticks, this.transformer);
        mouseMotionListenerImpl = new MouseMotionListenerImpl();
        this.axisChart.getChartSurface().addMouseMotionListener(mouseMotionListenerImpl);
    }

    protected abstract void prepareCursorLine(DataValue axisTick, Line2D.Double line);

    protected abstract int getPosition(MouseEvent e);

    protected abstract Stroke getStroke(Graphics2D gd);

    protected abstract Color getColor(Graphics2D gd);

    public void draw(Graphics2D gd) {
        if (mouseMotionListenerImpl.isVisible()) {
            axisTick = locationManager.nearestTick(mouseMotionListenerImpl.pos);
            if (axisTick != oldAxisTick) {
                prepareCursorLine(axisTick, cursorLine);
                oldAxisTick = axisTick;
                label = String.valueOf(axisTick);
            }
            Color color = gd.getColor();
            gd.setColor(getColor(gd));
            Stroke stroke = gd.getStroke();
            gd.setStroke(getStroke(gd));
            gd.draw(cursorLine);
            gd.setStroke(stroke);
            gd.setColor(color);
        }
    }

    protected class MouseMotionListenerImpl implements MouseMotionListener {

        public boolean overlap = true;
        public int pos;

        private void mouseMotion(MouseEvent e) {
            if (e.getX() < axisChart.getWidthEndOnSurface()
                    && e.getX() > axisChart.getLocation().x
                    && e.getY() < axisChart.getHeightEndOnSurface()
                    && e.getY() > axisChart.getLocation().y) {
                pos = CursorLocation.this.getPosition(e);
                overlap = true;
            } else {
                overlap = false;
            }
            axisChart.repaintChartSurface();
        }

        public void mouseDragged(MouseEvent e) {
            mouseMotion(e);
        }

        public final boolean isVisible() {
            return mouseMotionListenerImpl.overlap;
        }

        public void mouseMoved(MouseEvent e) {
            mouseMotion(e);
        }
    }
}
