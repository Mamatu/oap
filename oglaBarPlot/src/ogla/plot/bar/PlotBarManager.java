package ogla.plot.bar;

import ogla.core.util.Writer;
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
import java.awt.geom.Rectangle2D;
import java.awt.geom.Rectangle2D.Double;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JFrame;

public class PlotBarManager extends AxisChart2DPlotManager<Rectangle2D.Double> implements Properties {

    public Color getColor() {
        return colorOfFilling;
    }

    public Stroke getStroke() {
        return stroke;
    }
    private AxisChart2DPlotManagerBundle factory;
    private AxisChart2D axisChart = null;

    public PlotBarManager(AxisChart2D axisChart, AxisChart2DPlotManagerBundle factory) {
        this.factory = factory;
        this.axisChart = axisChart;
        configurationFrame = new ConfigurationFrame(axisChart, this, this);
    }

    @Override
    public void plotData(Displayer displayerInfo, DataBlock dataBlock, CoordinateTransformer coordinateXAxis, CoordinateTransformer coordinateYAxis, Graphics2D gd) {
        int originy = 0;
        if (originValue != null) {
            originy = coordinateYAxis.transform(axisChart, originValue).intValue();
        }
        if (originIndex != null) {
            originy = coordinateYAxis.transform(axisChart, dataBlock.get(originIndex,1).getNumber()).intValue();
        }
        Color defaultBackgroundColor = gd.getBackground();
        Color defaultColor = gd.getColor();
        Stroke defaultStroke = gd.getStroke();
        gd.setBackground(colorOfFilling);
        gd.setColor(colorOfEdge);
        gd.setStroke(stroke);
        for (int fa = getFirstPlotIndex(), shapeIndex = 0; fa + 1 < dataBlock.rows();
                fa++, shapeIndex++) {
            if (coordinateXAxis.inScope(axisChart, dataBlock.get(fa,0).getNumber())) {

                int x = coordinateXAxis.transform(axisChart, dataBlock.get(fa,0).getNumber()).intValue();
                int nx = axisChart.getLocation().x + axisChart.getWidth();
                if (coordinateXAxis.inScope(axisChart, dataBlock.get(fa + 1,0).getNumber())) {
                    nx = coordinateXAxis.transform(axisChart, dataBlock.get(fa + 1,0).getNumber()).intValue();
                }
                int w = nx - x;
                int y = coordinateYAxis.transform(axisChart, dataBlock.get(fa,1).getNumber()).intValue();
                if (y <= originy) {
                    if (y < axisChart.getLocation().y) {
                        y = axisChart.getLocation().y;
                    }
                    if (originy > axisChart.getHeightEndOnSurface()) {
                        originy = axisChart.getHeightEndOnSurface();
                    }
                    int h = originy - y;
                    this.shapes.get(shapeIndex).setFrame(x, y, w, h);
                } else {
                    if (originy < axisChart.getLocation().y) {
                        originy = axisChart.getLocation().y;
                    }
                    if (y > axisChart.getHeightEndOnSurface()) {
                        y = axisChart.getHeightEndOnSurface();
                    }
                    int h = y - originy;
                    this.shapes.get(shapeIndex).setFrame(x, originy, w, h);
                }

                gd.clearRect((int) this.shapes.get(shapeIndex).x, (int) this.shapes.get(shapeIndex).y,
                        (int) this.shapes.get(shapeIndex).width, (int) this.shapes.get(shapeIndex).height);
                gd.draw(this.shapes.get(shapeIndex));
            }
        }
        gd.setColor(defaultColor);
        gd.setBackground(defaultBackgroundColor);
        gd.setStroke(defaultStroke);
    }

    @Override
    public Double newShape() {
        return new Rectangle2D.Double();
    }

    @Override
    public String getLabel() {
        return "Bars";
    }
    private ConfigurationFrame configurationFrame = null;

    @Override
    public JFrame getFrame() {
        return configurationFrame;
    }
    private Color colorOfEdge = Color.black;
    private Color colorOfFilling = Color.gray;
    private BasicStroke stroke = new BasicStroke(1.f);

    public void setColorOfFilling(Color color) {
        colorOfFilling = color;
    }

    public void setColorOfEdge(Color color) {
        colorOfEdge = color;
    }
    private Float originValue = new Float(0.f);
    private Integer originIndex = null;

    public void setOriginValue(Float originValue) {
        this.originValue = originValue;
        originIndex = null;
    }

    public void setOriginIndex(Integer originIndex) {
        this.originIndex = originIndex;
        originValue = null;
    }

    public void setStroke(BasicStroke stroke) {
        this.stroke = stroke;
    }

    @Override
    public AxisChart2DPlotManagerBundle getPlotManagerFactory() {
        return factory;
    }

    @Override
    public byte[] save() {
        Writer writer = new Writer();
        writer.write(stroke.getLineWidth());
        writer.write(colorOfEdge.getRed());
        writer.write(colorOfEdge.getGreen());
        writer.write(colorOfEdge.getBlue());
        writer.write(colorOfFilling.getRed());
        writer.write(colorOfFilling.getGreen());
        writer.write(colorOfFilling.getBlue());
        if (originIndex == null) {
            writer.write(0);
            writer.write(originValue);
        } else if (originValue == null) {
            writer.write(1);
            writer.write(originIndex);
        }
        return writer.getBytes();
    }
    private Rectangle2D rectangle2D = new Rectangle2D.Float();

    @Override
    public void getGraphicSymbol(int x, int y, int width, int height, Graphics2D gd) {
        rectangle2D.setRect(x, y, width, height);
        Color defaultColor = gd.getColor();
        gd.setColor(colorOfFilling);
        gd.draw(rectangle2D);
        gd.setColor(defaultColor);
    }

    public String getDescription() {
        return "";
    }
}
