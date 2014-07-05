package ogla.plot.errorpoint;

import ogla.excore.Help;
import ogla.excore.util.Reader;
import ogla.excore.util.Reader.EndOfBufferException;
import ogla.axischart.AxisChart;
import ogla.axischart2D.AxisChart2D;
import ogla.axischart2D.plugin.plotmanager.AxisChart2DPlotManager;
import ogla.axischart2D.plugin.plotmanager.AxisChart2DPlotManagerBundle;
import java.awt.BasicStroke;
import java.awt.Color;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

public class PlotErrorPointManagerFactory extends AxisChart2DPlotManagerBundle {

    @Override
    public AxisChart2DPlotManager newPlotManager(AxisChart axisChart) {
        return new PlotErrorPointManager((AxisChart2D)axisChart, this);
    }

    @Override
    public String getLabel() {
        return "Error points";
    }

    @Override
    public String getSymbolicName() {
        return "plot_error_manager_factory_multichart";
    }

    @Override
    public boolean canBeSaved() {
        return true;
    }

    @Override
    public AxisChart2DPlotManager load(AxisChart2D axisChart, byte[] bytes) {
        Reader reader = new Reader(bytes);
        PlotErrorPointManager plotErrorPointManager = new PlotErrorPointManager(axisChart, this);
        try {
            final float size = reader.readFloat();
            final int r = reader.readInt();
            final int g = reader.readInt();
            final int b = reader.readInt();
            boolean isPercent = reader.readBoolean();
            plotErrorPointManager.xIsPercent(isPercent);
            int length = reader.readInt();
            float[] xError = new float[length];
            for (int fa = 0; fa < length; fa++) {
                xError[fa] = reader.readFloat();
            }
            isPercent = reader.readBoolean();
            plotErrorPointManager.yIsPercent(isPercent);
            length = reader.readInt();
            float[] yError = new float[length];
            for (int fa = 0; fa < length; fa++) {
                yError[fa] = reader.readFloat();
            }
            plotErrorPointManager.setColor(new Color(r, g, b));
            plotErrorPointManager.setStroke(new BasicStroke(size));
            plotErrorPointManager.setXError(xError);
            plotErrorPointManager.setYError(yError);
        } catch (EndOfBufferException ex) {
            Logger.getLogger(AxisChart2D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(AxisChart2D.class.getName()).log(Level.SEVERE, null, ex);
        }
        return plotErrorPointManager;
    }

    public Help getHelp() {
        return null;
    }

    public String getDescription() {
        return "";
    }
}
