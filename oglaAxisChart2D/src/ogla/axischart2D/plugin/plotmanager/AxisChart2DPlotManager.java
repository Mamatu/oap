package ogla.axischart2D.plugin.plotmanager;

import ogla.axischart.Displayer;
import ogla.axischart.plugin.plotmanager.PlotManager;
import ogla.axischart2D.DisplayerImpl;
import ogla.axischart2D.coordinates.CoordinateTransformer;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Stroke;
import java.util.ArrayList;
import javax.swing.JFrame;
import javax.swing.JPanel;
import ogla.core.data.DataBlock;
import ogla.core.data.DataRepository;

/**
 * Object of this class display on chart data in some form.
 * @param <T> class which identificy your shape which will be displayed on chart
 */
public abstract class AxisChart2DPlotManager<T> implements PlotManager {

    protected ArrayList<T> shapes = new ArrayList<T>();
    protected boolean isVisible = true;
    protected int firstIndex = 0;

    /**
     * Create new shape.
     * @return
     */
    public abstract T newShape();

    /**
     * Get color of displayed shapes.
     * @return color
     */
    public abstract Color getColor();

    /**
     * Get stroke which will be used during paint shapes.
     * @return stroke
     */
    public abstract Stroke getStroke();

    /**
     * Get PlotManagerFactory of this PlotManager
     * @return PlotManagerFactory
     */
    public abstract AxisChart2DPlotManagerBundle getPlotManagerFactory();

    /**
     * Data displayed by this manager is visible or not
     * @param b
     */
    public void setVisible(boolean b) {
        isVisible = b;
    }

    /**
     * Get configuration frame.
     * @return JFrame or null if this manager don't have configuration panel.
     */
    public abstract JFrame getFrame();

    public int getFirstPlotIndex() {
        return firstIndex;
    }

    private boolean checkOrder(DataBlock dataRepository) {
        if (dataRepository.rows() == 1) {
            return true;
        }
        for (int fa = 1; fa < dataRepository.rows(); fa++) {
            if (dataRepository.get(fa, 0).getNumber().floatValue() < dataRepository.get(fa - 1, 0).getNumber().floatValue()) {
                return false;
            }
        }
        return true;
    }

    private void adjust(DisplayerImpl displayer, DataBlock dataRepository) {
        firstIndex = 0;
        boolean stop = false;
        final int plotSize = dataRepository.rows();
        while (stop == false && dataRepository.get(firstIndex, 0).getNumber().floatValue()
                < displayer.getAxisChart().getXTicks().getLowestTick().floatValue()) {
            firstIndex++;
            if (firstIndex == dataRepository.rows() - 1) {
                stop = true;
            }
        }
        final int size = shapes.size();
        if (plotSize > size) {
            for (int fa = size; fa < plotSize; fa++) {
                this.shapes.add(newShape());
            }
        } else if (plotSize < size) {
            this.shapes.subList(plotSize, size).clear();
        }
    }

    public static class NoOrderException extends Exception {

        public NoOrderException(DataRepository dataRepository) {
            super("During draw this data set, order of x axis is not provided.\n"
                    + "It means that some x value is lower than some previous x value.\n"
                    + "Plugin which was used to loading data, don't support ascending order of x values.\n"
                    + "You should check/improve your data source.\n"
                    + "This error was detected in repository with label: " + dataRepository.getLabel());
        }
    }

    public void plot(DisplayerImpl displayer, DataBlock dataBlock,
            CoordinateTransformer coordinateXAxis, CoordinateTransformer coordinateYAxis,
            Graphics2D gd) throws NoOrderException {
        if (!checkOrder(dataBlock)) {
           // throw new NoOrderException(dataBlock.getDataRepository());
        }
        if (!isVisible || dataBlock.rows() <= 0) {
            return;
        }
        Color defaultColor = gd.getColor();
        Stroke defaultStroke = gd.getStroke();
        adjust(displayer, dataBlock);
        plotData(displayer, dataBlock, coordinateXAxis, coordinateYAxis, gd);
        gd.setStroke(defaultStroke);
        gd.setColor(defaultColor);
    }

    @Override
    public String toString() {
        return this.getLabel();
    }

    /**
     * Get label of this plot manager.
     * @return
     */
    public abstract String getLabel();

    /**
     * Plot data.
     * @param displayerInfo
     * @param repositoryData
     * @param coordinateXAxis
     * @param coordinateYAxis
     * @param gd
     */
    public abstract void plotData(Displayer displayerInfo, DataBlock dataRepository, CoordinateTransformer coordinateXAxis, CoordinateTransformer coordinateYAxis, Graphics2D gd);

    /**
     *
     * This method is used by AxisChart to save some information of plot manager
     * to file before exit of application.
     * @return array of bytes in which is stored state of object.
     */
    public abstract byte[] save();
}
