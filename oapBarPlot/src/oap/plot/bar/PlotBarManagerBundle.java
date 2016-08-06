package ogla.plot.bar;

import ogla.core.Help;
import ogla.core.util.Reader;
import ogla.axischart.AxisChart;
import ogla.axischart2D.AxisChart2D;
import ogla.axischart2D.plugin.plotmanager.AxisChart2DPlotManager;
import ogla.axischart2D.plugin.plotmanager.AxisChart2DPlotManagerBundle;
import java.awt.BasicStroke;
import java.awt.Color;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author marcin
 */
public class PlotBarManagerBundle extends AxisChart2DPlotManagerBundle {

    @Override
    public AxisChart2DPlotManager newPlotManager(AxisChart axisChart) {
        return new PlotBarManager((AxisChart2D) axisChart, this);
    }

    @Override
    public String getLabel() {
        return "Bar";
    }

    @Override
    public String getSymbolicName() {
        return "bar_chart_plot_manager_factory_analysis";
    }

    @Override
    public boolean canBeSaved() {
        return true;
    }

    @Override
    public AxisChart2DPlotManager load(AxisChart2D axisChart, byte[] bytes) {
        PlotBarManager plotBarManager = new PlotBarManager(axisChart, this);
        Reader reader = new Reader(bytes);
        try {
            float f = 0.f;
            int i = 0;
            int size = reader.readInt();
            int r = reader.readInt();
            int g = reader.readInt();
            int b = reader.readInt();
            int r1 = reader.readInt();
            int g1 = reader.readInt();
            int b1 = reader.readInt();
            int z = reader.readInt();
            if (z == 0) {
                f = reader.readFloat();
            } else {
                i = reader.readInt();
            }
            plotBarManager.setStroke(new BasicStroke(size));
            plotBarManager.setColorOfEdge(new Color(r, g, b));
            plotBarManager.setColorOfFilling(new Color(r1, g1, b1));
            if (z == 0) {
                plotBarManager.setOriginValue(f);
            } else {
                plotBarManager.setOriginIndex(i);
            }

        } catch (Reader.EndOfBufferException ex) {
            Logger.getLogger(AxisChart2D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(AxisChart2D.class.getName()).log(Level.SEVERE, null, ex);
        }
        return plotBarManager;
    }

    public Help getHelp() {
        return null;
    }

    public String getDescription() {
        return "";
    }
}
