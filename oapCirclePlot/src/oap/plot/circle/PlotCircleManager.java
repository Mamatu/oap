
package ogla.plot.circle;

import java.awt.geom.Rectangle2D;
import ogla.core.data.DataBlock;
import ogla.axischart.Displayer;
import ogla.axischart2D.AxisChart2D;
import ogla.axischart2D.plugin.plotmanager.AxisChart2DPlotManager;
import ogla.axischart2D.plugin.plotmanager.AxisChart2DPlotManagerBundle;
import ogla.axischart2D.coordinates.CoordinateTransformer;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Stroke;
import java.awt.geom.Ellipse2D;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JFrame;
import javax.swing.JPanel;
import ogla.core.util.Writer;

public class PlotCircleManager extends AxisChart2DPlotManager<Ellipse2D.Double> implements Properties {

    private AxisChart2DPlotManagerBundle factory;
    private Ellipse2D.Float point = new Ellipse2D.Float();

    @Override
    public void getGraphicSymbol(int x, int y, int width, int height, Graphics2D gd) {
        Color defaultColor = gd.getColor();
        Stroke defaultStroke = gd.getStroke();

        gd.setStroke(plotStroke);
        gd.setColor(plotColor);

        double dx = (double) x + (double) width / 2.0 - 0.1;
        double dy = (double) y + (double) height / 2.0 - 0.1;
        point.setFrame(dx, dy, 0.2d, 0.2d);
        gd.draw(point);

        gd.setStroke(defaultStroke);
        gd.setColor(defaultColor);
    }

    public AxisChart2DPlotManagerBundle getPlotManagerFactory() {
        return factory;
    }

    public Color getColor() {
        return plotColor;
    }

    public Stroke getStroke() {
        return plotStroke;
    }

    public PlotCircleManager(AxisChart2D axisChart, AxisChart2DPlotManagerBundle factory) {
        this.axisChart = axisChart;
        configurationPanel = new ConfigurationFrame(axisChart, this, this);
        this.factory = factory;
    }
    private AxisChart2D axisChart;
    private int px;
    private int py;

    @Override
    public void plotData(Displayer displayerInfo, DataBlock dataBlock, CoordinateTransformer coordinateXAxis, CoordinateTransformer coordinateYAxis, Graphics2D gd) {
        gd.setStroke(plotStroke);
        gd.setColor(plotColor);
        for (int shapeIndex = 0, fa = getFirstPlotIndex(); fa < dataBlock.rows(); fa++, shapeIndex++) {
            final boolean arg2 = coordinateXAxis.inScope(axisChart, dataBlock.get(fa,0).getNumber());
            final boolean arg3 = coordinateYAxis.inScope(axisChart, dataBlock.get(fa,1).getNumber());

            if (arg2 && arg3) {
                px = coordinateXAxis.transform(axisChart, dataBlock.get(fa,0).getNumber()).intValue();
                py = coordinateYAxis.transform(axisChart, dataBlock.get(fa,1).getNumber()).intValue();
                this.shapes.get(shapeIndex).x = px - 0.1d;
                this.shapes.get(shapeIndex).y = py - 0.1d;
                this.shapes.get(shapeIndex).width = .2d;
                this.shapes.get(shapeIndex).height = .2d;
                gd.draw(this.shapes.get(shapeIndex));
            }
        }
    }

    @Override
    public Ellipse2D.Double newShape() {
        return new Ellipse2D.Double();
    }

    @Override
    public String getLabel() {
        return "Points";
    }

    public void setColor(Color color) {
        this.plotColor = color;
    }

    public void setStroke(BasicStroke stroke) {
        this.plotStroke = stroke;
    }
    private BasicStroke plotStroke = new BasicStroke(1);
    private Color plotColor = Color.black;
    private ConfigurationFrame configurationPanel = null;

    @Override
    public JFrame getFrame() {
        return configurationPanel;
    }

    @Override
    public byte[] save() {
        Writer writer = new Writer();
        BasicStroke s = plotStroke;
        writer.write(s.getLineWidth());
        writer.write(plotColor.getRed());
        writer.write(plotColor.getGreen());
        writer.write(plotColor.getBlue());
        return writer.getBytes();
    }

    public String getDescription() {
        return "";
    }
}
