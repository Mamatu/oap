package ogla.grid;

import ogla.core.Help;
import ogla.core.util.Reader;
import ogla.core.util.Reader.EndOfBufferException;
import ogla.core.util.Writer;
import ogla.axischart2D.AxisChart2D;
import ogla.axischart2D.plugin.chartcomponent.AxisChart2DComponentBundle;
import ogla.axischart2D.plugin.chartcomponent.AxisChart2DComponent;
import ogla.chart.DrawingInfo;
import ogla.grid.gui.MainFrame;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.geom.Line2D;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JFrame;

public class Grid extends AxisChart2DComponent {

    private String label = "Grid";

    public String getLabel() {
        return label;
    }

    public String getDescription() {
        return "";
    }
    protected GridVertical gridVertical;
    protected GridHorizontal gridHorizontal;
    protected MainFrame mainFrame;

    public Grid(GridHorizontal gridHorizontal, GridVertical gridVertical, AxisChart2DComponentBundle bundleChartComponent, int index) {
        super(bundleChartComponent);
        this.label = "Grid " + String.valueOf(index);
        this.gridHorizontal = gridHorizontal;
        this.gridVertical = gridVertical;
        line1 = new Line2D.Double();
        line2 = new Line2D.Double();
        this.mainFrame = new MainFrame(gridHorizontal.getGridManager(), gridVertical.getGridManager());
    }

    public Grid(AxisChart2DComponentBundle bundleChartComponent, int index) {
        this(new GridHorizontal(bundleChartComponent, index), new GridVertical(bundleChartComponent, index), bundleChartComponent, index);
    }
    private Line2D.Double line1;
    private Line2D.Double line2;
    private AxisChart2D axisChart;

    @Override
    public void attachTo(AxisChart2D axisChart) {
        this.gridHorizontal.attachTo(axisChart);
        this.gridVertical.attachTo(axisChart);
        this.axisChart = axisChart;
    }

    @Override
    public void drawOnChart(Graphics2D gd, DrawingInfo drawingType) {
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
        gridHorizontal.drawOnChart(gd, drawingType);
        gridVertical.drawOnChart(gd, drawingType);
        super.drawOnChart(gd,drawingType);
    }

    @Override
    public JFrame getFrame() {
        return mainFrame;
    }

    @Override
    public byte[] save() {
        Writer writer = new Writer();
        byte[] a1 = gridVertical.save();
        byte[] a2 = gridHorizontal.save();
        writer.write(a1.length);
        writer.write(a1);
        writer.write(a2.length);
        writer.write(a2);
        return writer.getBytes();
    }

    @Override
    public void drawImpl(Graphics2D gd,DrawingInfo drawingType) {
    }

    public static class AxisChart2DComponentBundleImpl implements AxisChart2DComponentBundle {

        private int index = 0;

        public int getIndex() {
            int out = index;
            index++;
            return out;
        }

        public String getLabel() {
            return "Grid";
        }

        public String getDescription() {
            return "";
        }

        public boolean canBeSaved() {
            return true;
        }

        public AxisChart2DComponent newAxisChart2DComponent() {
            return new Grid(this, getIndex());
        }

        public String getSymbolicName() {
            return "grid_analysis";
        }

        public AxisChart2DComponent load(byte[] bytes) {
            Reader reader = new Reader(bytes);
            GridVertical.AxisChart2DComponentBundleImpl bundleVertical = new GridVertical.AxisChart2DComponentBundleImpl();
            GridHorizontal.AxisChart2DComponentBundleImpl bundleHorizontal = new GridHorizontal.AxisChart2DComponentBundleImpl();
            GridVertical gridVertical = null;
            GridHorizontal gridHorizontal = null;
            try {
                int length = reader.readInt();
                byte[] a1 = new byte[length];
                reader.readBytes(a1);
                length = reader.readInt();
                byte[] a2 = new byte[length];
                reader.readBytes(a2);
                gridVertical = (GridVertical) bundleVertical.load(a1);
                gridHorizontal = (GridHorizontal) bundleHorizontal.load(a2);
            } catch (EndOfBufferException ex) {
                Logger.getLogger(Grid.class.getName()).log(Level.SEVERE, null, ex);
            } catch (IOException ex) {
                Logger.getLogger(Grid.class.getName()).log(Level.SEVERE, null, ex);
            }
            return new Grid(gridHorizontal, gridVertical, this, getIndex());
        }

        public Help getHelp() {
            return null;
        }
    }
}
