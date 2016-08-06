package ogla.axischart2D;

import ogla.core.data.DataRepository;
import ogla.axischart2D.util.AxisIndices.OutOfRangeException;
import ogla.chart.DrawableOnChart;
import ogla.core.data.DataBlock;
import ogla.core.data.DataValue;
import ogla.axischart.Displayer;
import ogla.axischart2D.plugin.plotmanager.AxisChart2DPlotManager;
import ogla.axischart2D.coordinates.CoordinateTransformer;
import ogla.axischart2D.util.AxisIndices;
import ogla.chart.DrawingInfo;
import java.awt.Graphics2D;
import java.awt.Stroke;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JOptionPane;

public abstract class DisplayerImpl implements Displayer, DrawableOnChart {

    protected AxisChart2DPlotManager plotManager;
    protected AxisChart2D axisChart = null;
    protected Stroke stroke;

    public final AxisChart2DPlotManager getPlotManager() {
        return plotManager;
    }

    public AxisChart2D getAxisChart() {
        return axisChart;
    }

    public Stroke getStroke() {
        return stroke;
    }

    public DataRepository getDataRepository() {
        if (dataRepositories.length == 0) {
            return null;
        }
        return dataRepositories[0];
    }

    public int getNumberOfDataRepositories() {
        return dataRepositories.length;
    }

    public DataRepository getDataRepository(int index) {
        return dataRepositories[index];
    }
    private String name;

    public String getLabel() {
        return name;
    }
    protected DataRepository[] dataRepositories;
    protected DataRepository[] drawnDataRepositories;
    private AxisIndices axisIndices = new AxisIndicesImpl();

    private class AxisIndicesImpl extends AxisIndices {

        @Override
        public void setIndexX(int index) {
            super.setIndexX(index);
            for (DataBlockProxy dataBlockProxy : drawnDataBlockProxies) {
                dataBlockProxy.update(axisIndices);
            }

        }

        @Override
        public void setIndexY(int index) {
            super.setIndexY(index);
            for (DataBlockProxy dataBlockProxy : drawnDataBlockProxies) {
                dataBlockProxy.update(axisIndices);
            }

        }

        @Override
        public void setSideX(int type) {
            super.setSideX(type);
            for (DataBlockProxy dataBlockProxy : drawnDataBlockProxies) {
                dataBlockProxy.update(axisIndices);
            }
        }

        @Override
        public void setSideY(int type) {
            super.setSideY(type);
            for (DataBlockProxy dataBlockProxy : drawnDataBlockProxies) {
                dataBlockProxy.update(axisIndices);
            }
        }
    }

    public AxisIndices getAxisIndices() {
        return axisIndices;
    }

    public DisplayerImpl(AxisChart2D axisChart, DataRepository dataRepositoriesGroup, String name, AxisIndices axisIndices) {
        this.dataRepositories = new DataRepository[1];
        this.dataRepositories[0] = dataRepositoriesGroup;
        this.drawnDataRepositories = new DataRepository[1];
        this.drawnDataRepositories[0] = dataRepositoriesGroup;
        this.axisChart = axisChart;
        this.name = name;
        axisIndices.giveTo(this.axisIndices);
        fillDataBlockProxies();
    }

    private void fillDataBlockProxies() {
        for (DataRepository dataRepository : drawnDataRepositories) {
            for (int fa = 0; fa < dataRepository.size(); fa++) {
                if (dataRepository.get(fa) != null) {
                    drawnDataBlockProxies.add(new DataBlockProxy(dataRepository.get(fa), this.axisIndices));
                }
            }
        }
    }
    private List<DataBlockProxy> drawnDataBlockProxies = new ArrayList<DataBlockProxy>();

    public DisplayerImpl(AxisChart2D axisChart, DataRepository[] dra, String name, AxisIndices axisIndices) {
        this.dataRepositories = new DataRepository[dra.length];
        this.drawnDataRepositories = new DataRepository[dra.length];
        System.arraycopy(dra, 0, this.dataRepositories, 0, dra.length);
        System.arraycopy(dra, 0, this.drawnDataRepositories, 0, dra.length);
        this.axisChart = axisChart;
        this.name = name;
        axisIndices.giveTo(this.axisIndices);
        fillDataBlockProxies();

    }
    private boolean isVisible = true;

    public boolean isVisible() {
        return isVisible;
    }

    public void setVisible(boolean b) {
        isVisible = b;
        getPlotManager().setVisible(b);
    }

    private class DataBlockProxy implements DataBlock {

        private DataBlock dataBlock;
        private int indexx = 0;
        private int indexy = 1;

        public void update(AxisIndices axisIndices) {
            indexx = axisIndices.getXIndex(dataBlock);
            indexy = axisIndices.getYIndex(dataBlock);
        }

        public DataBlockProxy(DataBlock dataBlock, AxisIndices axisIndices) {
            this.dataBlock = dataBlock;
            indexx = axisIndices.getXIndex(dataBlock);
            indexy = axisIndices.getYIndex(dataBlock);
        }

        public DataValue get(int row, int column) {
            int c = column;
            if (column == 0) {
                c = indexx;
            } else if (column == 1) {
                c = indexy;
            }
            return dataBlock.get(row, c);
        }

        public int columns() {
            return dataBlock.columns() + 2;
        }

        public int rows() {
            return dataBlock.rows();
        }

        public DataRepository getDataRepository() {
            return dataBlock.getDataRepository();
        }
    }

    public void plotData(CoordinateTransformer coordinateXAxis, CoordinateTransformer cooridnateYAxis, Graphics2D gd) throws AxisChart2DPlotManager.NoOrderException {
        for (DataBlock dataBlock : drawnDataBlockProxies) {
            try {
                getPlotManager().plot(this, dataBlock, coordinateXAxis, cooridnateYAxis, gd);
            } catch (AxisChart2DPlotManager.NoOrderException noOrderException) {
                throw noOrderException;
            }
        }
    }

    public void drawOnChart(Graphics2D gd, DrawingInfo drawingInfo) {
        try {
            plotData(axisChart.getTransformerOfXAxis(), axisChart.getTransformerOfYAxis(), gd);
        } catch (AxisChart2DPlotManager.NoOrderException noOrderException) {
            JOptionPane.showMessageDialog(null, noOrderException.getMessage(), "Error in your data source", JOptionPane.ERROR_MESSAGE);
        }
    }
}
