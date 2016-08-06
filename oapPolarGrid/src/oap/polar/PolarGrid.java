package ogla.polar;

import ogla.excore.util.Writer;
import ogla.exdata.DataValue;
import ogla.axischart2D.AxisChart2D;
import ogla.axischart2D.Ticks;
import ogla.axischart2D.plugin.chartcomponent.AxisChart2DComponent;
import ogla.axischart2D.plugin.chartcomponent.AxisChart2DComponentBundle;
import ogla.chart.DrawingInfo;
import ogla.math.Vector3;
import ogla.math.Vector3.InvalidMatrixDimensionException;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Stroke;
import java.awt.geom.Line2D;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class PolarGrid extends AxisChart2DComponent {

    private String label = "Polar Grid";

    public String getLabel() {
        return label;
    }

    public String getDescription() {
        return "";
    }
    protected static int AXIS_X = 0;
    protected static int AXIS_Y = 1;
    protected AxisChart2D axisChart;
    private ConfigurationFrame configurationFrame = null;
    protected int gap = 0;
    protected boolean gridIsVisible = true;
    protected boolean textIsVisible = true;
    protected boolean linesAreVisible = true;
    private Color gridColor = Color.BLACK;
    private Color textColor = Color.BLACK;
    private double angleOfLine = -3.14d / 180.d * 20.d;
    private int angleToSave = 20;
    float[][] rotationLinesMatrix = {{(float) Math.cos(angleOfLine), -(float) Math.sin(angleOfLine), 0},
        {(float) Math.sin(angleOfLine), (float) Math.cos(angleOfLine), 0}, {0, 0, 1}};

    public void setAngleOfLines(double angleOfLine) {
        angleToSave = (int) angleOfLine;
        this.angleOfLine = -angleOfLine * 3.14d / 180.d;
        rotationLinesMatrix[0][0] = (float) Math.cos(this.angleOfLine);
        rotationLinesMatrix[0][1] = -(float) Math.sin(this.angleOfLine);
        rotationLinesMatrix[0][2] = 0;

        rotationLinesMatrix[1][0] = (float) Math.sin(this.angleOfLine);
        rotationLinesMatrix[1][1] = (float) Math.cos(this.angleOfLine);
        rotationLinesMatrix[1][2] = 0;

        rotationLinesMatrix[2][0] = 0;
        rotationLinesMatrix[2][1] = 0;
        rotationLinesMatrix[2][2] = 1;
    }

    protected void setGridColor(Color color) {
        this.gridColor = color;
    }

    protected void setTextColor(Color color) {
        this.textColor = color;
    }

    public PolarGrid(AxisChart2DComponentBundle bundleChartComponent, int index) {
        super(bundleChartComponent);
        configurationFrame = new ConfigurationFrame(this);
        this.label = "Polar Grid " + String.valueOf(index);
    }

    @Override
    public void attachTo(AxisChart2D axisChart) {
        if (axisChart == null) {
            return;
        }
        this.axisChart = axisChart;

        int maxX = axisChart.getTransformerOfXAxis().transform(axisChart, axisChart.getXTicks().getGreatestTick()).intValue();
        int minX = axisChart.getTransformerOfXAxis().transform(axisChart, axisChart.getXTicks().getLowestTick()).intValue();
        int minY = axisChart.getTransformerOfYAxis().transform(axisChart, axisChart.getYTicks().getGreatestTick()).intValue();
        int maxY = axisChart.getTransformerOfYAxis().transform(axisChart, axisChart.getYTicks().getLowestTick()).intValue();
        axisChart.repaintChartSurface();
    }

    @Override
    public byte[] save() {
        Writer writer = new Writer();
        try {
            writer.write(angleToSave);
            writer.write(gap);
            writer.write(gridColor.getRGB());
            writer.write(textColor.getRGB());
            writer.write(linesAreVisible);
            writer.write(gridIsVisible);
            writer.write(textIsVisible);
            if (isLeftCorner) {
                writer.write(0);
            } else if (isTick) {
                writer.write(1);
                writer.write(tickx.floatValue());
                writer.write(ticky.floatValue());
            } else if (isPixel) {
                writer.write(2);
                writer.write(pixelx);
                writer.write(pixely);
            }
            writer.write(usedAxis);
        } catch (IOException ex) {
            Logger.getLogger(PolarGrid.class.getName()).log(Level.SEVERE, null, ex);
        }
        return writer.getBytes();
    }

    @Override
    public JFrame getFrame() {
        return configurationFrame;
    }
    protected int usedAxis = AXIS_X;
    private double angle = -3.14d / 180.d;
    private Vector3 vec = new Vector3();
    private final float[][] rotationMatrix = {{(float) Math.cos(angle), -(float) Math.sin(angle), 0},
        {(float) Math.sin(angle), (float) Math.cos(angle), 0}, {0, 0, 1}};
    int xCenter;
    int yCenter;
    private Color defaultColor;
    protected boolean isLeftCorner = false;
    protected boolean isTick = true;
    protected Number tickx = 0;
    protected Number ticky = 0;
    protected boolean isPixel = false;
    protected int pixelx = 0;
    protected int pixely = 0;

    @Override
    public void drawImpl(Graphics2D gd, DrawingInfo drawingInfo) {
        int maxX = axisChart.getTransformerOfXAxis().transform(axisChart, axisChart.getXTicks().getGreatestTick()).intValue();
        int minX = axisChart.getTransformerOfXAxis().transform(axisChart, axisChart.getXTicks().getLowestTick()).intValue();
        int minY = axisChart.getTransformerOfYAxis().transform(axisChart, axisChart.getYTicks().getGreatestTick()).intValue();
        int maxY = axisChart.getTransformerOfYAxis().transform(axisChart, axisChart.getYTicks().getLowestTick()).intValue();

        if (isLeftCorner) {
            xCenter = minX;
            yCenter = maxY;
        } else if (isTick) {
            xCenter = axisChart.getTransformerOfXAxis().transform(axisChart, tickx).intValue();
            yCenter = axisChart.getTransformerOfYAxis().transform(axisChart, ticky).intValue();
        } else if (isPixel) {
            xCenter = pixelx;
            yCenter = pixely;
        }


        Ticks ticks = axisChart.getXTicks();
        if (usedAxis == AXIS_X) {
            ticks = axisChart.getXTicks();
        } else if (usedAxis == AXIS_Y) {
            ticks = axisChart.getYTicks();
        }
        defaultColor = gd.getColor();
        gd.setColor(gridColor);
        if (gridIsVisible) {
            drawCircles(ticks, minX, maxX, minY, maxY, xCenter, yCenter, gd);
        }
        if (linesAreVisible || textIsVisible) {
            drawLines(gd, minX, maxX, minY, maxY, xCenter, yCenter);
        }
        gd.setColor(defaultColor);
    }
    private Line2D.Float line = new Line2D.Float();
    private Line2D.Float angleLine = new Line2D.Float();
    private int[] temp = new int[2];
    private int[] outcomes = new int[4];
    private Stroke stroke = new BasicStroke(1.f);

    public void setStroke(Stroke stroke) {
        this.stroke = stroke;
    }

    private void drawLines(Graphics2D gd, int minX, int maxX, int minY, int maxY, int xCenter, int yCenter) {

        Vector3 vector = new Vector3();
        vector.set(maxX, 0, 0);
        double counter = 0;
        double angleOfTextSlope = 0;
        Stroke defaultStroke = gd.getStroke();

        gd.setStroke(stroke);
        boolean isFirst = true;
        try {
            while (counter < 2.d * Math.PI && counter > -2.d * Math.PI) {
                Vector3 base = new Vector3();
                base.set(vector);
                base.add(xCenter, yCenter, 0);

                int px = (int) xCenter;
                int py = (int) yCenter;
                int ppx = (int) base.x;
                int ppy = (int) base.y;

                int[] noutcomes = adjustToBorders(px, py, ppx, ppy, minX, maxX, minY, maxY, temp, outcomes);

                if (noutcomes != null) {
                    px = noutcomes[0];
                    py = noutcomes[1];
                    ppx = noutcomes[2];
                    ppy = noutcomes[3];

                    angleLine.x1 = px;
                    angleLine.x2 = ppx;
                    angleLine.y1 = py;
                    angleLine.y2 = ppy;

                    float xdiff = (float) (ppx - px) / 2.f;
                    float ydiff = (float) (py - ppy) / 2.f;

                    if (px != ppx || py != ppy) {
                        double a = -angleOfTextSlope * 180.d / 3.14d;
                        if (a >= 360.d) {
                            a = 0.d;
                        }
                        String angleStr = String.valueOf(a);
                        if (linesAreVisible && !isFirst) {
                            gd.draw(angleLine);
                        }
                        gd.translate(xCenter + xdiff, yCenter - ydiff);
                        gd.rotate(angleOfTextSlope);
                        Color gdLocalColor = gd.getColor();
                        gd.setColor(textColor);
                        if (textIsVisible) {
                            if (xCenter + xdiff > minX && xCenter + xdiff < maxX && yCenter - ydiff > minY
                                    && yCenter - ydiff < maxY && !isFirst) {
                                gd.drawString(angleStr.substring(0, getPointPosition(angleStr)), 0, 0);
                            }
                        }
                        gd.setColor(gdLocalColor);
                        gd.rotate(-angleOfTextSlope);
                        gd.translate(-(xCenter + xdiff), -(yCenter - ydiff));
                        angleOfTextSlope += angleOfLine;
                    }
                }

                counter += angleOfLine;
                vector = vector.rotate(rotationLinesMatrix, vector);
                isFirst = false;
            }
        } catch (InvalidMatrixDimensionException ex) {
            Logger.getLogger(PolarGrid.class.getName()).log(Level.SEVERE, null, ex);
        } finally {
            gd.setStroke(defaultStroke);
        }

    }

    private int getPointPosition(String text) {
        for (int fa = 0; fa < text.length(); fa++) {
            if (text.charAt(fa) == '.') {
                return fa;
            }
        }
        return text.length();
    }

    private void drawCircles(Ticks ticks, int minX, int maxX, int minY, int maxY, int xCenter, int yCenter, Graphics2D gd) {
        for (int fa = 0; fa < ticks.getSizeOfList(); fa++) {
            int px = 0;
            int py = 0;
            int ppx = 0;
            int ppy = 0;
            try {
                DataValue tick = ticks.getDataValue(fa);
                if (usedAxis == AXIS_X) {
                    py = yCenter;
                    px = axisChart.getTransformerOfXAxis().transform(axisChart, tick.getNumber()).intValue();
                    //vec = new Vector(xCenter+ppx, yCenter, 0);
                } else if (usedAxis == AXIS_Y) {
                    px = xCenter;
                    py = axisChart.getTransformerOfYAxis().transform(axisChart, tick.getNumber()).intValue();
                }

                vec.set(px - xCenter, py - yCenter, 0);
                for (int fb = 0; fb < 360; fb++) {
                    ppx = px;
                    ppy = py;
                    vec = vec.rotate(rotationMatrix, vec);
                    px = (int) vec.x + xCenter;
                    py = (int) vec.y + yCenter;
                    int counterX = 0;
                    int counterY = 0;
                    if (px >= minX && px <= maxX) {
                        counterX++;
                    }
                    if (ppx >= minX && ppx <= maxX) {
                        counterX++;
                    }

                    if (py >= minY && py <= maxY) {
                        counterY++;
                    }

                    if (ppy >= minY && ppy <= maxY) {
                        counterY++;
                    }

                    if (counterX >= 1 && counterY >= 1 && (gap == 0 || fb % gap == 0)) {
                        int[] noutcomes = adjustToBorders(px, py, ppx, ppy, minX, maxX, minY, maxY, temp, outcomes);
                        if (noutcomes != null) {
                            line.x1 = noutcomes[0];
                            line.x2 = noutcomes[2];
                            line.y1 = noutcomes[1];
                            line.y2 = noutcomes[3];
                            gd.setColor(Color.darkGray);
                            gd.draw(line);
                        }
                    }

                }

            } catch (InvalidMatrixDimensionException ex) {
                Logger.getLogger(PolarGrid.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }

    private int[] adjustToBorders(int px, int py, int ppx, int ppy, int minX, int maxX, int minY,
            int maxY, int[] temp, int[] outcomes) {
        float a = (float) (ppy - py) / (float) (ppx - px);
        float b = ppy - (float) ppx * a;

        int npx = px;
        int npy = py;
        int nppx = ppx;
        int nppy = ppy;

        if (npx > maxX || nppx > maxX) {
            float y = maxX * a + b;
            if (y >= minY && y <= maxY) {
                if (npx > maxX) {
                    npx = maxX;
                    npy = (int) y;
                }
                if (nppx > maxX) {
                    nppx = maxX;
                    nppy = (int) y;
                }
            }
        }

        if (npx < minX || nppx < minX) {
            float y = minX * a + b;
            if (y >= minY && y <= maxY) {
                if (npx < minX) {
                    npx = minX;
                    npy = (int) y;
                }
                if (nppx < minX) {
                    nppx = minX;
                    nppy = (int) y;
                }
            }
        }

        if (npy > maxY || nppy > maxY) {
            float x = (maxY - b) / a;
            if (x >= minX && x <= maxX) {
                if (npy > maxY) {
                    npy = maxY;
                    npx = (int) x;
                }
                if (nppy > maxY) {
                    nppy = maxY;
                    nppx = (int) x;
                }
            }
        }

        if (npy < minY || nppy < minY) {
            float x = (minY - b) / a;
            if (x >= minX && x <= maxX) {
                if (npy < minY) {
                    npy = minY;
                    npx = (int) x;
                }

                if (nppy < minY) {
                    nppy = minY;
                    nppx = (int) x;
                }
            }
        }
        if (npx > maxX || nppx > maxX) {
            return null;
        }
        if (npx < minX || nppx < minX) {
            return null;
        }
        if (npy > maxY || nppy > maxY) {
            return null;
        }
        if (npy < minY || nppy < minY) {
            return null;
        }


        outcomes[0] = npx;
        outcomes[1] = npy;
        outcomes[2] = nppx;
        outcomes[3] = nppy;

        return outcomes;
    }
}
