package ogla.plot.point;

import ogla.excore.util.Writer;
import java.awt.geom.Rectangle2D;
import ogla.exdata.DataBlock;
import ogla.axischart.Displayer;
import ogla.axischart2D.AxisChart2D;
import ogla.axischart2D.plugin.plotmanager.AxisChart2DPlotManager;
import ogla.axischart2D.plugin.plotmanager.AxisChart2DPlotManagerBundle;
import ogla.axischart2D.coordinates.CoordinateTransformer;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Stroke;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JFrame;

public class PlotPointManager extends AxisChart2DPlotManager<Rectangle2D.Double> implements Properties {

    private static final int XIndex = 0;
    private static final int YIndex = 1;
    private AxisChart2DPlotManagerBundle factory;
    private Rectangle2D.Float point = new Rectangle2D.Float();

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

    public PlotPointManager(AxisChart2D axisChart, AxisChart2DPlotManagerBundle factory) {
        this.axisChart = axisChart;
        configurationFrame = new ConfigurationFrame(axisChart, this, this);
        this.factory = factory;
    }
    private AxisChart2D axisChart;
    private int px;
    private int py;

    @Override
    public void plotData(Displayer displayer, DataBlock repositoryData, CoordinateTransformer coordinateXAxis, CoordinateTransformer coordinateYAxis, Graphics2D gd) {
        gd.setStroke(plotStroke);
        gd.setColor(plotColor);
        for (int shapeIndex = 0, fa = getFirstPlotIndex(); fa < repositoryData.rows(); fa++, shapeIndex++) {
            final boolean arg2 = coordinateXAxis.inScope(axisChart, repositoryData.get(fa,this.XIndex).getNumber());
            final boolean arg3 = coordinateYAxis.inScope(axisChart, repositoryData.get(fa,this.YIndex).getNumber());

            if (arg2 && arg3) {
                px = coordinateXAxis.transform(axisChart, repositoryData.get(fa,this.XIndex).getNumber()).intValue();
                py = coordinateYAxis.transform(axisChart, repositoryData.get(fa,this.YIndex).getNumber()).intValue();
                this.shapes.get(shapeIndex).x = px - 0.1d;
                this.shapes.get(shapeIndex).y = py - 0.1d;
                this.shapes.get(shapeIndex).width = .2d;
                this.shapes.get(shapeIndex).height = .2d;
                gd.draw(this.shapes.get(shapeIndex));
            }
        }
    }

    @Override
    public Rectangle2D.Double newShape() {
        return new Rectangle2D.Double();
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
    private ConfigurationFrame configurationFrame = null;

    @Override
    public JFrame getFrame() {
        return configurationFrame;
    }

    @Override
    public byte[] save() {
        Writer writer = new Writer();
        try {
            BasicStroke s = plotStroke;
            writer.write(s.getLineWidth());
            writer.write(plotColor.getRed());
            writer.write(plotColor.getGreen());
            writer.write(plotColor.getBlue());
        } catch (IOException ex) {
            Logger.getLogger(PlotPointManager.class.getName()).log(Level.SEVERE, null, ex);
        }
        return writer.getBytes();
    }
}
