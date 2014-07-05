package ogla.grid;

import ogla.core.Help;
import ogla.core.util.Reader;
import ogla.core.util.Reader.EndOfBufferException;
import ogla.core.util.Writer;
import ogla.core.data.DataValue;
import ogla.axischart2D.AxisChart2D;
import ogla.axischart2D.plugin.chartcomponent.AxisChart2DComponentBundle;
import ogla.axischart2D.plugin.chartcomponent.AxisChart2DComponent;
import ogla.chart.DrawingInfo;
import ogla.grid.gui.GridPanel;
import ogla.grid.gui.HorizontalFrame;
import ogla.grid.util.CursorLocation;
import ogla.grid.util.GridManager;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Stroke;
import java.awt.event.MouseEvent;
import java.awt.geom.Line2D;
import java.awt.geom.Line2D.Double;
import java.io.IOException;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JFrame;

/**
 *
 * @author marcin
 */
public class GridHorizontal extends GridComponent {

    private String label = "Grid Horizontal";

    public String getLabel() {
        return label;
    }

    public String getDescription() {
        return "";
    }
    protected HorizontalFrame horizontalFrame = null;
    protected Color loadedColor = null;

    @Override
    public JFrame getFrame() {
        return horizontalFrame;
    }

    @Override
    public byte[] save() {
        Writer writer = null;
        GridPanel gridPanel = horizontalFrame.getGridPanel();
        Color scolor = gridHorizontalManager.getColor();
        writer = new Writer();
        writer.write(gridPanel.dashed.length);
        for (int fa = 0; fa < gridPanel.dashed.length; fa++) {
            writer.write(gridPanel.dashed[fa]);
        }
        writer.write(gridPanel.phase);
        writer.write(gridPanel.width);
        writer.write(scolor.getRed());
        writer.write(scolor.getGreen());
        writer.write(scolor.getBlue());
        return writer.getBytes();
    }

    GridManager getGridManager() {
        return gridHorizontalManager;
    }

    private class GridHorizontalManager extends GridManager {

        public GridHorizontalManager(AxisChart2D axisChart) {
            super(axisChart, axisChart.getYTicks(), axisChart.getTransformerOfYAxis());
            this.setLength(20);
        }

        @Override
        public Number getPosition(int index) {
            double height = axisChart.getHeight();
            double size = 10.d;
            double step = height / size;
            return (int) ((step * index)) + axisChart.getLocation().y;
        }

        @Override
        public void drawLine(Graphics2D gd, Double line, int position) {
            line.y1 = position;
            line.y2 = position;
            line.x1 = axisChart.getLocation().x;
            line.x2 = axisChart.getWidthEndOnSurface();
            gd.draw(line);
        }

        @Override
        public void drawTick(Graphics2D gd, DataValue tick, int position) {
            String label = tick.getLabel();
            int x = GridHorizontal.this.axisChart.getLocation().x - label.length() * font.getSize();
            int y = position + font.getSize() / 2;
            Color dcolor = gd.getColor();
            gd.setColor(Color.black);
            gd.drawString(label, x, y);
            gd.rotate(0);
            gd.setColor(dcolor);
        }

        @Override
        protected Stroke getStroke(Graphics2D gd) {
            return basicStroke;
        }

        @Override
        protected boolean overlapCondition(DataValue tick1, Number pos1, DataValue tick2, Number pos2, List<DataValue> values, List<Number> positions) {
            final float tpos1 = pos1.floatValue();
            final float tpos2 = pos2.floatValue();
            final float fontSize = this.fontSize();
            return ((tpos1 + fontSize >= tpos2 && tpos1 + fontSize <= tpos2 + fontSize));
        }
    }
    private GridHorizontalManager gridHorizontalManager;
    private CursorLocation cursorLocation;
    private AxisChart2D axisChart;

    @Override
    public void attachTo(AxisChart2D axisChart) {
        this.axisChart = axisChart;
        if (axisChart != null) {
            gridHorizontalManager = new GridHorizontalManager(axisChart);
            horizontalFrame.getGridPanel().setGridManager(gridHorizontalManager);
            cursorLocation = new CursorLocation(axisChart, axisChart.getYTicks(), axisChart.getTransformerOfYAxis()) {

                private BasicStroke basicStroke = new BasicStroke(1);

                @Override
                protected void prepareCursorLine(DataValue axisTick, Double line) {
                    line.x1 = axisChart.getWidthEndOnSurface();
                    line.x2 = axisChart.getLocation().x;
                    int y = transformer.transform(axisChart, axisTick.getNumber().floatValue()).intValue();
                    line.y1 = y;
                    line.y2 = y;
                }

                @Override
                protected int getPosition(MouseEvent e) {
                    return e.getY();
                }

                @Override
                protected Stroke getStroke(Graphics2D gd) {
                    return basicStroke;
                }

                @Override
                protected Color getColor(Graphics2D gd) {
                    return Color.RED;
                }
            };
            if (loadedColor != null) {
                gridHorizontalManager.setColor(loadedColor);
            } else {
                gridHorizontalManager.setColor(Color.black);
            }
        }
    }

    public GridHorizontal(AxisChart2DComponentBundle bundleChartComponent, int index) {
        super(bundleChartComponent);
        this.label = "Grid Horizontal " + String.valueOf(index);
        horizontalFrame = new HorizontalFrame();
    }
    private Line2D.Double borderLine = new Line2D.Double();

    public void drawImpl(Graphics2D gd, DrawingInfo drawingType) {
        borderLine.x1 = axisChart.getLocation().x;
        borderLine.x2 = axisChart.getLocation().x;
        borderLine.y1 = axisChart.getLocation().y;
        borderLine.y2 = axisChart.getHeightEndOnSurface();
        gd.draw(borderLine);
        gridHorizontalManager.manage(gd);
        if (DrawingInfo.DrawingOnChart == drawingType) {
            cursorLocation.draw(gd);
        }
    }

    public static class AxisChart2DComponentBundleImpl implements AxisChart2DComponentBundle {

        private int index = 0;

        public int getIndex() {
            int out = index;
            index++;
            return out;
        }

        public String getLabel() {
            return "Grid horizontal";
        }

        public String getDescription() {
            return "";
        }

        public boolean canBeSaved() {
            return true;
        }

        public AxisChart2DComponent newAxisChart2DComponent() {
            GridHorizontal gridHorizontal = new GridHorizontal(this, getIndex());
            float[] a = {0.5f};
            gridHorizontal.horizontalFrame.getGridPanel().createStroke(1.f, a, 0.f, 1.f);
            gridHorizontal.loadedColor = new Color(256, 256, 256);
            return gridHorizontal;
        }

        public String getSymbolicName() {
            return "grid_horizontal_analysis";
        }

        public AxisChart2DComponent load(byte[] bytes) {
            GridHorizontal gridHorizontal = new GridHorizontal(this, getIndex());
            try {

                Reader reader = new Reader(bytes);
                int length = reader.readInt();
                float[] dashed = new float[length];
                for (int fa = 0; fa < length; fa++) {
                    dashed[fa] = reader.readFloat();
                }
                float phase = reader.readFloat();
                float width = reader.readFloat();
                int r = reader.readInt();
                int g = reader.readInt();
                int b = reader.readInt();
                gridHorizontal.horizontalFrame.getGridPanel().createStroke(1.f, dashed, phase, width);
                gridHorizontal.loadedColor = new Color(r, g, b);

            } catch (EndOfBufferException ex) {
                Logger.getLogger(GridVertical.class.getName()).log(Level.SEVERE, null, ex);
            } catch (IOException ex) {
                Logger.getLogger(GridVertical.class.getName()).log(Level.SEVERE, null, ex);
            }
            return gridHorizontal;
        }

        public Help getHelp() {
            return null;
        }
    }
}
