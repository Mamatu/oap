package ogla.axis;

import ogla.core.Help;
import ogla.core.util.Reader;
import ogla.core.util.Reader.EndOfBufferException;
import ogla.core.util.Writer;
import ogla.axischart2D.AxisChart2D;
import ogla.axischart2D.plugin.chartcomponent.AxisChart2DComponentBundle;
import ogla.axischart2D.plugin.chartcomponent.AxisChart2DComponent;
import ogla.axis.gui.AxisFrame;
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

public class AxisX extends AxisComponent {

    private String label = "Axis X";

    public String getLabel() {
        return label;
    }

    public String getDescription() {
        return "";
    }
    protected AxisFrame xFrame = new AxisFrame();
    protected Color loadedColor = null;

    @Override
    public JFrame getFrame() {
        return xFrame;
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

    private class AxisHorizontalManager extends AxisManager {

        public Border border;

        public AxisHorizontalManager(AxisChart2D axisChart) {
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
        private Line2D.Double lineOfTick = new Line2D.Double();

        @Override
        public void drawLine(Graphics2D gd, Double line, int position) {
            if (this.drawWholeLines) {
                line.x1 = position;
                line.x2 = position;
                line.y1 = axisChart.getLocation().y;
                line.y2 = axisChart.getHeightEndOnSurface();
                gd.draw(line);
            } else {
                int height = axisChart.getHeight();
                height = height * 5 / 300;
                if (mainLineCenterOfWindow) {
                    lineOfTick.y1 = axisChart.getHeightEndOnSurface() / 2.d - height;
                    lineOfTick.y2 = axisChart.getHeightEndOnSurface() / 2.d + height;
                } else if (mainLinePixel) {
                    lineOfTick.y1 = pixel - height;
                    lineOfTick.y2 = pixel + height;
                } else if (mainLineTick) {
                    int tempposition = axisChart.getTransformerOfYAxis().transform(axisChart, tick).intValue();
                    lineOfTick.y1 = tempposition - height;
                    lineOfTick.y2 = tempposition + height;
                }
                lineOfTick.x1 = position;
                lineOfTick.x2 = position;
                gd.draw(lineOfTick);

            }
        }

        @Override
        public void drawTick(Graphics2D gd, DataValue tick, int position) {
            String label = tick.getLabel();
            float x = (float) position - ((float) label.length() / 2.f);
            int y = AxisX.this.getAxisChart2D().getHeightEndOnSurface() + font.getSize();
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
        protected boolean overlapCondition(DataValue tick1, Number pos1, DataValue tick2, Number pos2) {
            String label1 = tick1.getLabel();
            String label2 = tick2.getLabel();
            final float tpos1 = pos1.floatValue();
            final float tpos2 = pos2.floatValue();
            final float fontSize = this.fontSize();
            final float ln1 = label1.length() * fontSize;
            final float ln2 = label2.length() * fontSize;
            return ((tpos1 + ln1 >= tpos2 && tpos1 + ln1 <= tpos2 + ln2));
        }

        @Override
        public void prepareMainLineForCenterOfWindow(Float mainLine) {
            mainLine.x1 = axisChart.getLocation().x;
            mainLine.x2 = axisChart.getWidthEndOnSurface();
            mainLine.y1 = axisChart.getHeightEndOnSurface() / 2.f;
            mainLine.y2 = axisChart.getHeightEndOnSurface() / 2.f;
        }

        @Override
        public void prepareMainLineForPixel(Float mainLine) {
            mainLine.x1 = axisChart.getLocation().x;
            mainLine.x2 = axisChart.getWidthEndOnSurface();
            mainLine.y1 = this.pixel;
            mainLine.y2 = this.pixel;
        }

        @Override
        public void prepareMainLineForTick(Float mainLine) {
            float v = axisChart.getTransformerOfYAxis().transform(axisChart, tick).floatValue();
            mainLine.x1 = axisChart.getLocation().x;
            mainLine.x2 = axisChart.getWidthEndOnSurface();
            mainLine.y1 = v;
            mainLine.y2 = v;
        }
    }
    private AxisHorizontalManager axisManagerImpl;
    private CursorLocation cursorLocation;

    @Override
    public void attachTo(AxisChart2D axisChart2D) {
        super.attachTo(axisChart2D);
        if (axisChart2D != null) {
            axisManagerImpl = new AxisHorizontalManager(axisChart2D);
            xFrame.setAxisManager(axisManagerImpl);
            cursorLocation = new CursorLocation(axisChart2D, axisChart2D.getXTicks(), axisChart2D.getTransformerOfXAxis()) {

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
                axisManagerImpl.setColor(loadedColor);
            }
        }
    }

    public AxisX(AxisChart2DComponentBundle bundleChartComponent, int index) {
        super(bundleChartComponent);
        label = "Axis X " + String.valueOf(index);
    }

    public void drawImpl(Graphics2D gd, DrawingInfo drawingType) {
        axisManagerImpl.manage(gd);
        if (drawingType == DrawingInfo.DrawingOnChart) {
            cursorLocation.draw(gd);
        }
    }

    public static class AxisChart2DComponentBundleImpl implements AxisChart2DComponentBundle {

        public Help getHelp() {
            return null;
        }

        public String getLabel() {
            return "Axis vertical";
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
            AxisX axisX = new AxisX(this, getIndex());
            float[] a = {0.5f};
            axisX.xFrame.getAxisPanel().createStroke(1.f, a, 0.f, 1.f);
            axisX.loadedColor = new Color(255, 255, 255);
            return axisX;
        }

        public String getSymbolicName() {
            return "axis_vertical_analysis";
        }

        public boolean isObjectOfThisBundle(AxisChart2DComponent axisChartComponent) {
            return axisChartComponent instanceof AxisX;
        }

        public AxisChart2DComponent load(byte[] bytes) {
            AxisX axis = new AxisX(this, getIndex());
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
