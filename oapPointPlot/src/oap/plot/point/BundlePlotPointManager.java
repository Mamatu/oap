package ogla.plot.point;

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

public class BundlePlotPointManager extends AxisChart2DPlotManagerBundle {

    @Override
    public AxisChart2DPlotManager newPlotManager(AxisChart axisChart) {
        return new PlotPointManager((AxisChart2D) axisChart, this);
    }

    @Override
    public String getLabel() {
        return "Points";
    }

    @Override
    public String getSymbolicName() {
        return "plot_point_manager_factory_multichart";
    }

    @Override
    public boolean canBeSaved() {
        return true;
    }

    @Override
    public AxisChart2DPlotManager load(AxisChart2D axisChart, byte[] bytes) {
        PlotPointManager plotPointManager = null;
        try {
            Reader reader = new Reader(bytes);
            final float size = reader.readFloat();
            final int r = reader.readInt();
            final int g = reader.readInt();
            final int b = reader.readInt();
            plotPointManager = new PlotPointManager(axisChart, this);
            plotPointManager.setStroke(new BasicStroke(size));
            Color color = new Color(r, g, b);
            plotPointManager.setColor(color);
        } catch (EndOfBufferException ex) {
            Logger.getLogger(AxisChart2D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(AxisChart2D.class.getName()).log(Level.SEVERE, null, ex);
        }
        return plotPointManager;
    }

    public Help getHelp() {
        return null;
    }

    public String getDescription() {
        return "";
    }
}
