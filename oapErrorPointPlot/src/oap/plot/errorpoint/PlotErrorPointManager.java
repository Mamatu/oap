package ogla.plot.errorpoint;

import ogla.excore.util.Writer;
import ogla.exdata.DataBlock;
import ogla.axischart.Displayer;
import ogla.axischart2D.AxisChart2D;
import ogla.axischart2D.DisplayerImpl;
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
import javax.swing.JPanel;

public class PlotErrorPointManager extends AxisChart2DPlotManager<ErrorPoint> implements Properties {

    private AxisChart2DPlotManagerBundle factory;

    public AxisChart2DPlotManagerBundle getPlotManagerFactory() {
        return factory;
    }
    private ErrorPoint errorPoint = new ErrorPoint();

    public void getGraphicSymbol(int x, int y, int w, int h, Graphics2D gd) {
        Color defaultColor = gd.getColor();
        Stroke defaultStroke = gd.getStroke();
        errorPoint.point.setFrame(x + w / 2, y + h / 2, 0.1, 0.1);
        errorPoint.lineH.setLine(x, y + h / 2, x + w, y + h / 2);
        errorPoint.lineV.setLine(x + w / 2, y, x + w / 2, y + h);

        gd.setColor(plotColor);
        gd.setStroke(plotStroke);
        gd.draw(errorPoint.point);
        gd.draw(errorPoint.lineV);
        gd.draw(errorPoint.lineH);

        gd.setColor(defaultColor);
        gd.setStroke(defaultStroke);
    }

    public PlotErrorPointManager(AxisChart2D axisChart, AxisChart2DPlotManagerBundle factory) {
        this.axisChart = axisChart;
        configurationFrame = new ConfigurationFrame(axisChart, this, this);
        this.factory = factory;
    }
    private AxisChart2D axisChart;
    private int px;
    private int py;
    private float lH;
    private float mlH;
    private float lV;
    private float mlV;

    private Number getValue(int index, DataBlock dataBlock, int fa) {
        if (dataBlock.rows() <= fa) {
            return 0;
        }
        if (dataBlock.columns() <= index) {
            return 0;
        }
        return dataBlock.get(fa,index).getNumber();
    }

    private int getXErrorLength(float error) {
        float l = axisChart.getXTicks().getGreatestTick().floatValue() - axisChart.getXTicks().getLowestTick().floatValue();
        float w = axisChart.getWidth();
        return (int) ((error * w) / l);
    }

    private int getXErrorLengthPercent(float percent, float value) {
        percent = percent / 100.f;
        value = value * percent;
        float l = axisChart.getXTicks().getGreatestTick().floatValue() - axisChart.getXTicks().getLowestTick().floatValue();
        float w = axisChart.getWidth();
        return (int) ((value * w) / l);
    }

    private int getYErrorLength(float error) {
        float l = axisChart.getYTicks().getGreatestTick().floatValue() - axisChart.getYTicks().getLowestTick().floatValue();
        float h = axisChart.getHeight();
        return (int) ((error * h) / l);
    }

    private int getYErrorLengthPercent(float percent, float value) {
        percent = percent / 100.f;
        value = value * percent;
        float l = axisChart.getXTicks().getGreatestTick().floatValue() - axisChart.getXTicks().getLowestTick().floatValue();
        float h = axisChart.getHeight();
        return (int) ((value * h) / l);
    }

    @Override
    public void plotData(Displayer displayer, DataBlock dataBlock, CoordinateTransformer coordinateXAxis, CoordinateTransformer coordinateYAxis, Graphics2D gd) {
        gd.setColor(plotColor);
        gd.setStroke(plotStroke);
        int indexX = 0;
        int indexY = 1;
        int errorX = configurationFrame.getIndexErrorX();
        int errorY = configurationFrame.getIndexErrorY();
        for (int shapeIndex = 0, fa = 0; fa < dataBlock.rows(); fa++, shapeIndex++) {
            final boolean arg2 = coordinateXAxis.inScope(axisChart, dataBlock.get(fa,indexX).getNumber());
            final boolean arg3 = coordinateYAxis.inScope(axisChart, dataBlock.get(fa,indexY).getNumber());
            if (arg2 && arg3) {
                px = coordinateXAxis.transform(axisChart, dataBlock.get(fa,indexX).getNumber()).intValue();
                py = coordinateYAxis.transform(axisChart, dataBlock.get(fa,indexY).getNumber()).intValue();
                if (!configurationFrame.isNumberX()) {
                    lH = (float) coordinateXAxis.transform(axisChart, dataBlock.get(fa,indexX).getNumber().doubleValue() + getValue(errorX, dataBlock, fa).doubleValue()).floatValue();
                    mlH = (float) coordinateXAxis.transform(axisChart, dataBlock.get(fa,indexX).getNumber().doubleValue() - getValue(errorX, dataBlock, fa).doubleValue()).floatValue();
                } else {
                    lH = (float) coordinateXAxis.transform(axisChart, dataBlock.get(fa,indexX).getNumber().doubleValue() + errorX).floatValue();
                    mlH = (float) coordinateXAxis.transform(axisChart, dataBlock.get(fa,indexX).getNumber().doubleValue() - errorX).floatValue();
                }
                if (!configurationFrame.isNumberY()) {
                    lV = (float) coordinateYAxis.transform(axisChart, dataBlock.get(fa,indexY).getNumber().doubleValue() + getValue(errorY, dataBlock, fa).doubleValue()).floatValue();
                    mlV = (float) coordinateYAxis.transform(axisChart, dataBlock.get(fa,indexY).getNumber().doubleValue() - getValue(errorY, dataBlock, fa).doubleValue()).floatValue();
                } else {
                    lV = (float) coordinateYAxis.transform(axisChart, dataBlock.get(fa,indexY).getNumber().doubleValue() + errorY).floatValue();
                    mlV = (float) coordinateYAxis.transform(axisChart, dataBlock.get(fa,indexY).getNumber().doubleValue() - errorY).floatValue();
                }
                this.shapes.get(shapeIndex).point.x = px - 0.1d;
                this.shapes.get(shapeIndex).point.y = py - 0.1d;
                this.shapes.get(shapeIndex).point.width = .2d;
                this.shapes.get(shapeIndex).point.height = .2d;

                this.shapes.get(shapeIndex).lineH.x1 = lH;
                if (this.shapes.get(shapeIndex).lineH.x1 > axisChart.getWidthEndOnSurface()) {
                    this.shapes.get(shapeIndex).lineH.x1 = axisChart.getWidthEndOnSurface();
                }
                this.shapes.get(shapeIndex).lineH.x2 = mlH;
                if (this.shapes.get(shapeIndex).lineH.x2 < axisChart.getLocation().x) {
                    this.shapes.get(shapeIndex).lineH.x2 = axisChart.getLocation().x;
                }
                this.shapes.get(shapeIndex).lineH.y1 = py;
                this.shapes.get(shapeIndex).lineH.y2 = py;

                this.shapes.get(shapeIndex).lineV.x1 = px;
                this.shapes.get(shapeIndex).lineV.x2 = px;
                this.shapes.get(shapeIndex).lineV.y1 = lV;
                if (this.shapes.get(shapeIndex).lineV.y1 > axisChart.getHeightEndOnSurface()) {
                    this.shapes.get(shapeIndex).lineV.y1 = axisChart.getHeightEndOnSurface();
                }
                this.shapes.get(shapeIndex).lineV.y2 = mlV;
                if (this.shapes.get(shapeIndex).lineV.y2 < axisChart.getLocation().y) {
                    this.shapes.get(shapeIndex).lineV.y2 = axisChart.getLocation().y;
                }
                gd.draw(this.shapes.get(shapeIndex).point);
                gd.draw(this.shapes.get(shapeIndex).lineH);
                gd.draw(this.shapes.get(shapeIndex).lineV);
            }
        }
    }

    @Override
    public ErrorPoint newShape() {
        return new ErrorPoint();
    }

    @Override
    public String getLabel() {
        return "Error points";
    }
    private BasicStroke plotStroke = new BasicStroke(1);
    private Color plotColor = Color.black;
    private ConfigurationFrame configurationFrame = null;
    private float[] xError = {0.f};
    private float[] yError = {0.f};

    @Override
    public JFrame getFrame() {
        return configurationFrame;
    }

    public Color getColor() {
        return plotColor;
    }

    public Stroke getStroke() {
        return plotStroke;
    }

    public void setColor(Color color) {
        this.plotColor = color;
    }

    public void setStroke(BasicStroke stroke) {
        plotStroke = stroke;
    }

    public void setXError(float[] x) {
        xError = x;
    }

    public void setYError(float[] y) {
        yError = y;
    }

    @Override
    public byte[] save() {
        Writer writer = new Writer();
        try {
            writer.write(plotStroke.getLineWidth());
            writer.write(plotColor.getRed());
            writer.write(plotColor.getGreen());
            writer.write(plotColor.getBlue());
            writer.write(this.xPercent);
            writer.write(xError.length);
            for (int fa = 0; fa < xError.length; fa++) {
                writer.write(xError[fa]);
            }
            writer.write(this.yPercent);
            writer.write(yError.length);
            for (int fa = 0; fa < yError.length; fa++) {
                writer.write(yError[fa]);
            }
        } catch (IOException ex) {
            Logger.getLogger(PlotErrorPointManager.class.getName()).log(Level.SEVERE, null, ex);
        }
        return writer.getBytes();
    }
    private boolean xPercent = false;

    public void xIsPercent(boolean b) {
        xPercent = b;
    }
    private boolean yPercent = false;

    public void yIsPercent(boolean b) {
        yPercent = b;
    }
    int xIndexOfColumnInRepository = -1;

    public void setXIndexOfColumnInRepository(int index) {
        xIndexOfColumnInRepository = index;
    }
    int yIndexOfColumnInRepository = -1;

    public void setYIndexOfColumnInRepository(int index) {
        yIndexOfColumnInRepository = index;
    }

    public String getDescription() {
        return "";
    }
}
