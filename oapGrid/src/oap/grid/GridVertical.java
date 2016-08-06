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
import ogla.grid.gui.VerticalFrame;
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
import javax.swing.JPanel;

/**
 *
 * @author marcin
 */
public class GridVertical extends GridComponent {

    private String label = "Grid Vertical";

    public String getLabel() {
        return label;
    }

    public String getDescription() {
        return "";
    }
    protected VerticalFrame verticalFrame;
    protected Color loadedColor = null;

    @Override
    public JFrame getFrame() {
        return verticalFrame;
    }

    @Override
    public byte[] save() {
        Writer writer = null;
        GridPanel gridPanel = verticalFrame.getGridPanel();
        Color scolor = gridVerticalManager.getColor();
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
        return gridVerticalManager;
    }

    private class GridVerticalManager extends GridManager {

        public Border border;

        public GridVerticalManager(AxisChart2D axisChart) {
            super(axisChart, axisChart.getXTicks(), axisChart.getTransformerOfXAxis());
            this.setLength(20);
        }

        @Override
        public Number getPosition(int index) {
            double width = axisChart.getWidth();
            double size = getLength();
            double step = width / size;
            return (step * index) + axisChart.getLocation().x;
        }

        @Override
        public void drawLine(Graphics2D gd, Double line, int position) {
            line.x1 = position;
            line.x2 = position;
            line.y1 = axisChart.getLocation().y;
            line.y2 = axisChart.getHeightEndOnSurface();
            gd.draw(line);
        }

        @Override
        public void drawTick(Graphics2D gd, DataValue tick, int position) {
            String label = tick.getLabel();
            float x = (float) position - ((float) label.length() / 2.f);
            int y = GridVertical.this.axisChart.getHeightEndOnSurface() + font.getSize();
            Color defaultColor = gd.getColor();
            gd.setColor(Color.black);
            gd.drawString(label, x, y);
            gd.rotate(0);
            gd.setColor(defaultColor);
        }

        @Override
        protected Stroke getStroke(Graphics2D gd) {
            return basicStroke;
        }

        @Override
        protected boolean overlapCondition(DataValue tick1, Number pos1, DataValue tick2, Number pos2, List<DataValue> values, List<Number> positions) {
            String label1 = tick1.getLabel();
            String label2 = tick2.getLabel();
            final float tpos1 = pos1.floatValue();
            final float tpos2 = pos2.floatValue();
            final float fontSize = this.fontSize();
            final float ln1 = label1.length() * fontSize;
            final float ln2 = label2.length() * fontSize;
            return ((tpos1 + ln1 >= tpos2 && tpos1 + ln1 <= tpos2 + ln2));
        }
    }
    private GridVerticalManager gridVerticalManager;
    private CursorLocation cursorLocation;
    private AxisChart2D axisChart = null;

    @Override
    public void attachTo(AxisChart2D axisChart) {
        this.axisChart = axisChart;
        if (axisChart != null) {
            gridVerticalManager = new GridVerticalManager(axisChart);
            verticalFrame.getGridPanel().setGridManager(gridVerticalManager);
            cursorLocation = new CursorLocation(axisChart, axisChart.getXTicks(), axisChart.getTransformerOfXAxis()) {

                private BasicStroke basicStroke = new BasicStroke(1);

                @Override
                protected void prepareCursorLine(DataValue axisTick, Double line) {
                    int x = transformer.transform(axisChart, axisTick.getNumber().floatValue()).intValue();
                    line.x1 = x;
                    line.x2 = x;
                    line.y1 = axisChart.getHeightEndOnSurface();
                    line.y2 = axisChart.getLocation().y;
                }

                @Override
                protected int getPosition(MouseEvent e) {
                    return e.getX();
                }

                @Override
                protected Stroke getStroke(Graphics2D gd) {
                    return basicStroke;
                }

                @Override
                protected Color getColor(Graphics2D gd) {
                    return Color.BLUE;
                }
            };
            if (loadedColor != null) {
                gridVerticalManager.setColor(loadedColor);
            }
        }
    }

    public GridVertical(AxisChart2DComponentBundle bundleChartComponent, int index) {
        super(bundleChartComponent);
        verticalFrame = new VerticalFrame();
        label = "Grid Vertical " + String.valueOf(index);
    }
    private Line2D.Double borderLine = new Line2D.Double();

    public void drawImpl(Graphics2D gd, DrawingInfo drawingType) {
        borderLine.x1 = axisChart.getLocation().x;
        borderLine.x2 = axisChart.getWidthEndOnSurface();
        borderLine.y1 = axisChart.getHeightEndOnSurface();
        borderLine.y2 = axisChart.getHeightEndOnSurface();
        gd.draw(borderLine);
        gridVerticalManager.manage(gd);
        if (drawingType == DrawingInfo.DrawingOnChart) {
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
            return "Grid vertical";
        }

        public String getDescription() {
            return "";
        }

        public boolean canBeSaved() {
            return true;
        }

        public AxisChart2DComponent newAxisChart2DComponent() {
            GridVertical gridVertical = new GridVertical(this, getIndex());
            float[] a = {0.5f};
            gridVertical.verticalFrame.getGridPanel().createStroke(1.f, a, 0.f, 1.f);
            gridVertical.loadedColor = new Color(255, 255, 255);
            return gridVertical;
        }

        public String getSymbolicName() {
            return "grid_vertical_analysis";
        }

        public boolean isObjectOfThisBundle(AxisChart2DComponent axisChartComponent) {
            return axisChartComponent instanceof GridVertical;
        }

        public AxisChart2DComponent load(byte[] bytes) {
            GridVertical gridVertical = new GridVertical(this, getIndex());
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
                gridVertical.verticalFrame.getGridPanel().createStroke(1.f, dashed, phase, width);
                gridVertical.loadedColor = new Color(r, g, b);

            } catch (EndOfBufferException ex) {
                Logger.getLogger(GridVertical.class.getName()).log(Level.SEVERE, null, ex);
            } catch (IOException ex) {
                Logger.getLogger(GridVertical.class.getName()).log(Level.SEVERE, null, ex);
            }
            return gridVertical;
        }

        public Help getHelp() {
            return null;
        }
    }
}
