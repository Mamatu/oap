
package ogla.plot.line;

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

/**
 *
 * @author marcin
 */
public class BundlePlotLineManager extends AxisChart2DPlotManagerBundle {

    @Override
    public AxisChart2DPlotManager newPlotManager(AxisChart axisChart) {
        return new PlotLineManager((AxisChart2D)axisChart, this);
    }

    @Override
    public String getLabel() {
        return "Lines";
    }

    @Override
    public String getSymbolicName() {
        return "plot_line_manager_factory_multichart";
    }

    @Override
    public boolean canBeSaved() {
        return true;
    }

    @Override
    public AxisChart2DPlotManager load(AxisChart2D axisChart, byte[] bytes) {
        PlotLineManager plotLineManager = null;
        try {
            Reader reader = new Reader(bytes);
            final float size = reader.readFloat();
            final int r = reader.readInt();
            final int g = reader.readInt();
            final int b = reader.readInt();
            plotLineManager = new PlotLineManager(axisChart, this);
            plotLineManager.setStroke(new BasicStroke(size));
            Color color = new Color(r, g, b);
            plotLineManager.setColor(color);
        } catch (EndOfBufferException ex) {
            Logger.getLogger(AxisChart2D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(AxisChart2D.class.getName()).log(Level.SEVERE, null, ex);
        }
        return plotLineManager;
    }

    public Help getHelp() {
        return null;
    }

    public String getDescription() {
        return "";
    }
}
