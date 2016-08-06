package ogla.axischart3D.util;

import ogla.data.DataRepository;
import ogla.axischart.Displayer;
import ogla.axischart.plugin.plotmanager.PlotManager;
import ogla.chart.DrawableOnChart;
import ogla.chart.DrawingInfo;
import java.awt.Color;
import java.awt.Graphics2D;
import java.util.ArrayList;
import java.util.List;

public class DisplayerImpl implements Displayer, DrawableOnChart {

    protected List<DataRepository> dataRepositories = new ArrayList<DataRepository>();
   
    private String label = "";
    private String descritpion = "";
    private boolean isVisible = true;
    private Color color = Color.black;

    public void setColor(Color color) {
        this.color = color;
    }

    public Color getColor() {
        return color;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public int getNumberOfDataRepositories() {
        return dataRepositories.size();
    }

    public DataRepository getDataRepository() {
        if (dataRepositories.size() == 0) {
            return null;
        }
        return dataRepositories.get(0);
    }

    public DataRepository getDataRepository(int index) {
        return dataRepositories.get(index);
    }

    public String getLabel() {
        return label;
    }

    public void setVisible(boolean b) {
        this.isVisible = b;
    }

    public boolean isVisible() {
        return isVisible;
    }

    public String getDescription() {
        return descritpion;
    }

    public void drawOnChart(Graphics2D gd, DrawingInfo drawingInfo) {
    }

    public PlotManager getPlotManager() {
        throw new UnsupportedOperationException("Not supported yet.");
    }
}
