package ogla.axis;

import ogla.core.Help;
import ogla.core.util.Reader;
import ogla.core.util.Reader.EndOfBufferException;
import ogla.core.util.Writer;
import ogla.axis.gui.AxisFrame;
import ogla.axischart2D.AxisChart2D;
import ogla.axischart2D.plugin.chartcomponent.AxisChart2DComponentBundle;
import ogla.axischart2D.plugin.chartcomponent.AxisChart2DComponent;
import ogla.axis.util.CursorLocation;
import ogla.axis.util.AxisManager;
import ogla.chart.DrawingInfo;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Stroke;
import java.awt.event.MouseEvent;
import java.awt.geom.Line2D;
import java.awt.geom.Line2D.Double;
import java.awt.geom.Line2D.Float;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JFrame;
import ogla.core.data.DataValue;

public class AxisY extends AxisComponent {

    private String label = "Axis Y";

    public String getLabel() {
        return label;
    }

    public String getDescription() {
        return "";
    }
    AxisFrame yFrame = new AxisFrame();
    protected Color loadedColor = null;

    @Override
    public JFrame getFrame() {
        return yFrame;
    }

    @Override
    public byte[] save() {
        Writer writer = null;
        Color scolor = axisManagerImpl.getColor();
        writer = new Writer();
        int i = axisManagerImpl.getTypeOfMainLine();
        writer.write(i);
        if (i == 0) {
            writer.write(0);
        } else if (i == 1) {
            writer.write(axisManagerImpl.getPixel());
        } else if (i == 2) {
            writer.write(axisManagerImpl.getTick().floatValue());
        }
        writer.write(scolor.getRed());
        writer.write(scolor.getGreen());
        writer.write(scolor.getBlue());
        return writer.getBytes();
    }

    AxisManager getAxisManager() {
        return axisManagerImpl;
    }

    public class AxisVerticalManager extends AxisManager {

        public AxisVerticalManager(AxisChart2D axisChart) {
            super(axisChart, axisChart.getYTicks(), axisChart.getTransformerOfYAxis());
            this.setLength(20);
        }

        @Override
        public Number getPosition(int index) {
            double height = axisChart.getHeight();
            double size = 10.d;
            double step = height / size;
            return (double) ((step * index)) + axisChart.getLocation().y;
        }

        @Override
        public void drawLine(Graphics2D gd, Double line, int position) {
            if (this.drawWholeLines) {
                line.y1 = position;
                line.y2 = position;
                line.x1 = axisChart.getLocation().x;
                line.x2 = axisChart.getWidthEndOnSurface();
                gd.draw(line);
            } else {
                int width = axisChart.getWidth();
                width = width * 5 / 300;

                if (mainLineCenterOfWindow) {
                    lineOfTick.x1 = axisChart.getLocation().x + axisChart.getWidth() / 2.d - width;
                    lineOfTick.x2 = axisChart.getLocation().x + axisChart.getWidth() / 2.d + width;
                } else if (mainLinePixel) {
                    lineOfTick.x1 = pixel - width;
                    lineOfTick.x2 = pixel + width;
                } else if (mainLineTick) {
                    int tempposition = axisChart.getTransformerOfXAxis().transform(axisChart, tick).intValue();
                    lineOfTick.x1 = tempposition - width;
                    lineOfTick.x2 = tempposition + width;
                }

                lineOfTick.y1 = position;
                lineOfTick.y2 = position;
                gd.draw(lineOfTick);
            }
        }
        private Line2D.Double lineOfTick = new Line2D.Double();

        @Override
        public void drawTick(Graphics2D gd, DataValue tick, int position) {
            String label = tick.getLabel();
            int x = AxisY.this.axisChart.getLocation().x - label.length() * font.getSize();
            int y = position + font.getSize() / 2;
            Color color = gd.getColor();
            gd.setColor(Color.black);
            gd.drawString(label, x, y);
            gd.rotate(0);
            gd.setColor(color);
        }

        @Override
        protected Stroke getStroke(Graphics2D gd) {
            return basicStroke;
        }

        @Override
        protected boolean overlapCondition(DataValue tick1, Number pos1, DataValue tick2, Number pos2) {
            final float tpos1 = pos1.floatValue();
            final float tpos2 = pos2.floatValue();
            final float fontSize = this.fontSize();
            return ((tpos1 + fontSize >= tpos2 && tpos1 + fontSize <= tpos2 + fontSize));
        }

        @Override
        public void prepareMainLineForCenterOfWindow(Float mainLine) {
            mainLine.x1 = axisChart.getLocation().x + axisChart.getWidth() / 2.f;
            mainLine.x2 = axisChart.getLocation().x + axisChart.getWidth() / 2.f;
            mainLine.y1 = axisChart.getLocation().y;
            mainLine.y2 = axisChart.getHeightEndOnSurface();
        }

        @Override
        public void prepareMainLineForPixel(Float mainLine) {
            mainLine.x1 = pixel;
            mainLine.x2 = pixel;
            mainLine.y1 = axisChart.getLocation().y;
            mainLine.y2 = axisChart.getHeightEndOnSurface();
        }

        @Override
        public void prepareMainLineForTick(Float mainLine) {
            float v = axisChart.getTransformerOfXAxis().transform(axisChart, tick).floatValue();
            mainLine.x1 = v;
            mainLine.x2 = v;
            mainLine.y1 = axisChart.getLocation().y;
            mainLine.y2 = axisChart.getHeightEndOnSurface();
        }
    }
    private AxisVerticalManager axisManagerImpl;
    private CursorLocation cursorLocation;
    private AxisChart2D axisChart;

    @Override
    public void attachTo(AxisChart2D axisChart) {
        this.axisChart = axisChart;
        if (axisChart != null) {
            axisManagerImpl = new AxisVerticalManager(axisChart);
            yFrame.setAxisManager(axisManagerImpl);
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
                axisManagerImpl.setColor(loadedColor);
            } else {
                axisManagerImpl.setColor(Color.black);
            }
        }
    }

    public AxisY(AxisChart2DComponentBundle bundleChartComponent, int index) {
        super(bundleChartComponent);
        label = "Axis Y " + String.valueOf(index);
    }

    public void drawImpl(Graphics2D gd, DrawingInfo drawingInfo) {
        axisManagerImpl.manage(gd);
        if (DrawingInfo.DrawingOnChart == drawingInfo) {
            cursorLocation.draw(gd);
        }
    }

    public static class AxisChart2DComponentBundleImpl implements AxisChart2DComponentBundle {

        public Help getHelp() {
            return null;
        }

        public String getLabel() {
            return "Axis horizontal";
        }

        public String getDescription() {
            return "";
        }

        public boolean canBeSaved() {
            return true;
        }
        private int index = 0;

        public int getIndex() {
            int out = index;
            index++;
            return out;
        }

        public AxisChart2DComponent newAxisChart2DComponent() {
            AxisY axisVertical = new AxisY(this, getIndex());
            float[] a = {0.5f};
            axisVertical.yFrame.getAxisPanel().createStroke(1.f, a, 0.f, 1.f);
            axisVertical.loadedColor = new Color(256, 256, 256);
            return axisVertical;
        }

        public String getSymbolicName() {
            return "axis_horizontal_analysis";
        }

        public AxisChart2DComponent load(byte[] bytes) {
            AxisY axis = new AxisY(this, getIndex());
            try {

                Reader reader = new Reader(bytes);
                int t = reader.readInt();
                if (t == 0) {
                    reader.readInt();
                    axis.axisManagerImpl.setTypeOfMainLine(t, null);
                } else if (t == 1) {
                    int p = reader.readInt();
                    axis.axisManagerImpl.setTypeOfMainLine(t, p);
                } else if (t == 2) {
                    float f = reader.readFloat();
                    axis.axisManagerImpl.setTypeOfMainLine(t, f);
                }
                int r = reader.readInt();
                int g = reader.readInt();
                int b = reader.readInt();
                axis.loadedColor = new Color(r, g, b);

            } catch (EndOfBufferException ex) {
                Logger.getLogger(AxisX.class.getName()).log(Level.SEVERE, null, ex);
            } catch (IOException ex) {
                Logger.getLogger(AxisX.class.getName()).log(Level.SEVERE, null, ex);
            }
            return axis;
        }
    }
}
