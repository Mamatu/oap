package ogla.chart;


import java.awt.Color;
import java.awt.Graphics2D;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import ogla.core.util.Reader;
import ogla.core.util.Writer;

/**
 * Represent chart on surface.
 */
public abstract class Chart implements DrawableOnChartSurface, DrawableOnChart {

    protected IOManager iOManager = new IOManager();

    private class WorkspaceProperties implements IOEntity {

        public void save(Writer writer) throws IOException {
            writer.write(width);
            writer.write(prevWidth);
            writer.write(height);
            writer.write(prevHeight);
            writer.write(prevX);
            writer.write(prevY);
        }

        public void load(Reader reader) throws IOException, Reader.EndOfBufferException {
            width = reader.readInt();
            prevWidth = reader.readInt();
            height = reader.readInt();
            prevHeight = reader.readInt();
            prevX = reader.readInt();
            prevY = reader.readInt();
        }
    }

    public byte[] save() {
        Writer writer = new Writer();
        try {
            for (int fa = 0; fa < iOManager.iOEntities.size(); fa++) {
                IOEntity iOEntity = iOManager.iOEntities.get(fa);
                iOEntity.save(writer);
            }
        } catch (Exception ex) {
            System.err.println(ex.getMessage());
        }
        return writer.getBytes();
    }

    public void load(byte[] bytes) {
        Reader reader = new Reader();
        reader.setBytes(bytes);
        try {
            for (int fa = 0; fa < iOManager.iOEntities.size(); fa++) {
                IOEntity iOEntity = iOManager.iOEntities.get(fa);
                iOEntity.load(reader);
            }
        } catch (Exception ex) {
            System.err.println(ex.getMessage());
        }
    }

    /**
     * Class defines stack of displayed objects. Place of displayed object on
     * chart is depended on index of this object in stack.
     */
    public static abstract class DisplayingStack {

        /**
         *
         * @return size of stack
         */
        protected abstract int size();

        /**
         *
         * @param index index of stack's cell
         * @return DrawableChart object at position of
         */
        protected abstract DrawableOnChart get(int index);
    }

    /**
     * Gets display stack
     *
     * @return
     */
    public abstract DisplayingStack getDisplayStack();
    /**
     * Reference to ChartSurface object.
     */
    protected ChartSurface chartSurface;
    /**
     * true - plotting is visible, false - it is invisible.
     */
    protected boolean isVisible = true;
    protected Color background = Color.white;

    /**
     * Default construct. {@link unregister} and {@link rebuild} methods.
     *
     * @param chartSurface ChartSurface object
     */
    public Chart(ChartSurface chartSurface) {
        this.chartSurface = chartSurface;
        this.chartSurface.getDrawable().add(this);
        iOManager.iOEntities.add(new WorkspaceProperties());
    }

    public void setBackground(Color color) {
        background = color;
    }

    public Color getBackground() {
        return background;
    }

    /**
     *
     * @return true - plotting is visible, false - it is invisible.
     */
    public boolean isVisible() {
        return isVisible;
    }

    /**
     *
     * @param b true - plotting is visible, false - it is invisible.
     */
    public void visible(boolean b) {
        isVisible = b;
    }

    /**
     * Unregistered plot from specially ChartSurface object. Chart object is
     * removed from list this type of object in ChartSurface object.
     *
     * @return true - if operation is done, false - otherwise
     */
    public void unregister() {
        chartSurface.getDrawable().remove(this);
        chartSurface.repaint();
        chartSurface = null;
    }

    /**
     * Repaint chartSurface which store this object.
     */
    public void repaintChartSurface() {
        chartSurface.repaint();
    }

    /**
     *
     * @return ChartSurface object
     */
    public ChartSurface getChartSurface() {
        return chartSurface;
    }
    private int width = 0;
    private int prevWidth = 0;
    private int height = 0;
    private int prevHeight = 0;
    private int prevX = 0;
    private int prevY = 0;

    /**
     * Refresh and adjust width of chart.
     */
    public void refreshWidth() {
        width = (int) ((float) (chartSurface.getWidth()) * 0.8);
        if (width != prevWidth) {
            for (int fa = 0; fa < getChartListeners().size(); fa++) {
                ChartListener chartListener = getChartListeners().get(fa);
                chartListener.isChangedSize(this);
            }
            prevWidth = width;
        }
    }

    /**
     *
     * @return width
     */
    public int getWidth() {
        return width;
    }

    /**
     * Refresh and adjust width of chart.
     */
    public void refreshHeight() {
        height = (int) ((float) (chartSurface.getHeight()) * 0.8);
        if (height != prevHeight) {
            for (int fa = 0; fa < getChartListeners().size(); fa++) {
                ChartListener chartListener = getChartListeners().get(fa);
                chartListener.isChangedSize(this);
            }
            prevHeight = height;
        }
    }

    /**
     *
     * @return height
     */
    public int getHeight() {
        return height;
    }
    private java.awt.Point location = new java.awt.Point();

    /**
     * Refresh location of chart.
     */
    public void refreshLocation() {
        int xoff = (int) ((float) (chartSurface.getWidth()) * 0.1);
        int yoff = (int) ((float) (chartSurface.getHeight()) * 0.1);
        location.x = chartSurface.getLocation().x + xoff;
        location.y = chartSurface.getLocation().y + yoff;
        if (location.x != prevX || location.y != prevY) {
            for (int fa = 0; fa < getChartListeners().size(); fa++) {
                ChartListener chartListener = getChartListeners().get(fa);
                chartListener.isChangedLocation(this);
            }
            prevX = location.x;
            prevY = location.y;
        }
    }

    /**
     * Return location.
     */
    public java.awt.Point getLocation() {
        return location;
    }

    /**
     *
     * @return return point which is end of chart's width.
     */
    public int getWidthEndOnSurface() {
        return getWidth() + getLocation().x;
    }

    /**
     *
     * @return return point which is end of chart's height.
     */
    public int getHeightEndOnSurface() {
        return getHeight() + getLocation().y;
    }

    /**
     * Class which listen change of chart.
     */
    public interface ChartListener {

        /**
         * Invoke when size is changed.
         *
         * @param chart
         */
        public void isChangedSize(Chart chart);

        /**
         * Invoke when location is changed.
         *
         * @param chart
         */
        public void isChangedLocation(Chart chart);
    }
    private ChartListenersList chartListenersList = new ChartListenersList();

    /**
     *
     * @return ChartListener object
     */
    public ChartListenersList getChartListeners() {
        return chartListenersList;
    }

    public class ChartListenersList {

        private List<ChartListener> list = new ArrayList<ChartListener>();

        protected ChartListener get(int index) {
            return list.get(index);
        }

        protected int size() {
            return list.size();
        }

        public void add(ChartListener chartListener) {
            list.add(chartListener);
        }

        public void remove(ChartListener chartListener) {
            list.remove(chartListener);
        }
    }

    public final void plotIt(Graphics2D gd, DrawingInfo drawingInfo) {
        if (isVisible()) {
        }
    }

    public void drawOnChart(Graphics2D gd, DrawingInfo drawingInfo) {
        plotIt(gd, drawingInfo);
    }

    public void drawOnChartSurface(Graphics2D gd, DrawingInfo drawingInfo) {
        refreshLocation();
        refreshWidth();
        refreshHeight();
        Color backColor = getBackground();
        gd.setBackground(backColor);
        gd.clearRect(getLocation().x, getLocation().y, getWidth(), getHeight());
        for (int fa = 0; fa < getDisplayStack().size(); fa++) {
            getDisplayStack().get(fa).drawOnChart(gd, drawingInfo);
        }
        this.drawOnChart(gd, drawingInfo);
    }
}
