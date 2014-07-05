package ogla.axis;

import ogla.core.Help;
import ogla.core.util.Reader;
import ogla.core.util.Reader.EndOfBufferException;
import ogla.core.util.Writer;
import ogla.axis.gui.MainFrame;
import ogla.axischart2D.plugin.chartcomponent.AxisChart2DComponentBundle;
import ogla.axischart2D.plugin.chartcomponent.AxisChart2DComponent;
import ogla.axischart2D.AxisChart2D;
import ogla.chart.DrawingInfo;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.geom.Line2D;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JFrame;

public class Axes extends AxisComponent {

    public String label = "Axes";

    public String getLabel() {
        return label;
    }

    public String getDescription() {
        return "";
    }
    protected AxisX axisX = null;
    protected AxisY axisY = null;
    protected MainFrame axesFrame;
    private AxisChart2D axisChart;

    @Override
    public void attachTo(AxisChart2D axisChart) {
        if (axisChart == null) {
            return;
        }
        if (this.axisX == null) {
            AxisChart2DComponentBundleImpl acdcbi = (Axes.AxisChart2DComponentBundleImpl) bundleChartComponent;
            this.axisX = new AxisX(bundleChartComponent, acdcbi.getIndex());
        }
        if (this.axisY == null) {
            AxisChart2DComponentBundleImpl acdcbi = (Axes.AxisChart2DComponentBundleImpl) bundleChartComponent;
            this.axisY = new AxisY(bundleChartComponent, acdcbi.getIndex());
        }
        this.axisX.attachTo(axisChart);
        this.axisY.attachTo(axisChart);
        this.axisChart = axisChart;
        line1 = new Line2D.Double();
        line2 = new Line2D.Double();
        axesFrame = new MainFrame(this.axisX.getAxisManager(), this.axisY.getAxisManager());

    }

    public Axes(AxisChart2DComponentBundle bundleChartComponent, int index) {
        super(bundleChartComponent);
        label = "Axes " + String.valueOf(index);
    }

    public Axes(AxisX axisX, AxisY axisY, AxisChart2DComponentBundle bundleChartComponent, int index) {
        super(bundleChartComponent);
        this.axisX = axisX;
        this.axisY = axisY;
        label = "Axes " + String.valueOf(index);
    }
    private Line2D.Double line1;
    private Line2D.Double line2;

    @Override
    public JFrame getFrame() {
        return axesFrame;
    }

    @Override
    public byte[] save() {
        Writer writer = new Writer();
        byte[] a1 = axisX.save();
        byte[] a2 = axisY.save();
        writer.write(a1.length);
        writer.write(a1);
        writer.write(a2.length);
        writer.write(a2);
        return writer.getBytes();
    }

    @Override
    public void drawImpl(Graphics2D gd, DrawingInfo drawingInfo) {
        line1.x1 = axisChart.getWidthEndOnSurface();
        line1.x2 = axisChart.getLocation().x;
        line1.y1 = axisChart.getLocation().y;
        line1.y2 = axisChart.getLocation().y;
        line2.x1 = axisChart.getWidthEndOnSurface();
        line2.x2 = axisChart.getWidthEndOnSurface();
        line2.y1 = axisChart.getLocation().y;
        line2.y2 = axisChart.getHeightEndOnSurface();
        gd.setBackground(Color.white);
        gd.draw(line1);
        gd.draw(line2);
        axisY.drawOnChart(gd, drawingInfo);
        axisX.drawOnChart(gd, drawingInfo);
    }

    public static class AxisChart2DComponentBundleImpl implements AxisChart2DComponentBundle {

        public String getLabel() {
            return "Axis";
        }

        public String getDescription() {
            return "";
        }

        public boolean canBeSaved() {
            return true;
        }

        public AxisChart2DComponent newAxisChart2DComponent() {
            return new Axes(this, getIndex());
        }

        public String getSymbolicName() {
            return "axis_analysis";
        }
        private int index = 0;

        public int getIndex() {
            int out = index;
            index++;
            return out;
        }

        public AxisChart2DComponent load(byte[] bytes) {
            Reader reader = new Reader(bytes);
            AxisX.AxisChart2DComponentBundleImpl bundleX = new AxisX.AxisChart2DComponentBundleImpl();
            AxisX axisX = null;
            AxisY.AxisChart2DComponentBundleImpl bundleY = new AxisY.AxisChart2DComponentBundleImpl();
            AxisY axisY = null;
            try {
                int length = reader.readInt();
                byte[] a1 = new byte[length];
                reader.readBytes(a1);
                length = reader.readInt();
                byte[] a2 = new byte[length];
                reader.readBytes(a2);
                axisX = (AxisX) bundleX.load(a1);
                axisY = (AxisY) bundleY.load(a2);
            } catch (EndOfBufferException ex) {
                Logger.getLogger(Axes.class.getName()).log(Level.SEVERE, null, ex);
            } catch (IOException ex) {
                Logger.getLogger(Axes.class.getName()).log(Level.SEVERE, null, ex);
            }
            return new Axes(axisX, axisY, this, getIndex());
        }

        public Help getHelp() {
            return null;
        }
    }
}
