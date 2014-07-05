package ogla.plot.line;

import ogla.excore.util.Writer;
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
import java.awt.geom.Line2D;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JFrame;

public class PlotLineManager extends AxisChart2DPlotManager<Line2D.Double> implements Properties {

    private int XIndex = 0;
    private int YIndex = 1;
    private AxisChart2DPlotManagerBundle factory;

    public Color getColor() {
        return plotColor;
    }

    public Stroke getStroke() {
        return plotStroke;
    }

    public AxisChart2DPlotManagerBundle getPlotManagerFactory() {
        return factory;
    }
    private AxisChart2D axisChart;
    private Line2D.Float line = new Line2D.Float();

    @Override
    public void getGraphicSymbol(int x, int y, int width, int height, Graphics2D gd) {
        Color defaultColor = gd.getColor();
        Stroke defaultStroke = gd.getStroke();

        gd.setStroke(plotStroke);
        gd.setColor(plotColor);

        line.setLine(x, y + height / 2, x + width, y + height / 2);
        gd.draw(line);

        gd.setStroke(defaultStroke);
        gd.setColor(defaultColor);
    }

    public PlotLineManager(AxisChart2D axisChart, AxisChart2DPlotManagerBundle factory) {
        this.axisChart = axisChart;
        configuraionFrame = new ConfigurationFrame(axisChart, this, this);
        this.factory = factory;
    }
    private int px = -1;
    private int py = -1;
    private int[] a = new int[2];
    private int[] outcomes = new int[4];

    @Override
    public void plotData(Displayer displayer, DataBlock dataBlock, CoordinateTransformer coordinateXAxis, CoordinateTransformer coordinateYAxis, Graphics2D gd) {
        XIndex = 0;
        YIndex = 1;
        gd.setStroke(plotStroke);
        gd.setColor(plotColor);
        int maxX = coordinateXAxis.transform(axisChart, axisChart.getXTicks().getGreatestTick()).intValue();
        int minX = coordinateXAxis.transform(axisChart, axisChart.getXTicks().getLowestTick()).intValue();
        int minY = coordinateYAxis.transform(axisChart, axisChart.getYTicks().getGreatestTick()).intValue();
        int maxY = coordinateYAxis.transform(axisChart, axisChart.getYTicks().getLowestTick()).intValue();
        Integer previous = null;
        for (int shapeIndex = 0, fa = 1; fa < dataBlock.rows(); fa++, shapeIndex++) {
            final boolean arg2 = coordinateXAxis.inScope(axisChart, dataBlock.get(fa, this.XIndex).getNumber());
            final boolean arg3 = coordinateYAxis.inScope(axisChart, dataBlock.get(fa, this.YIndex).getNumber());
            final boolean arg4 = coordinateXAxis.inScope(axisChart, dataBlock.get(fa - 1, this.XIndex).getNumber());
            final boolean arg5 = coordinateYAxis.inScope(axisChart, dataBlock.get(fa - 1, this.YIndex).getNumber());
            if ((arg2 || arg4) && (arg3 || arg5)) {
                int ppx;
                int ppy;
                if (fa == getFirstPlotIndex() + 1 || (previous == null || previous != fa - 1)) {
                    Number xnumber = dataBlock.get(fa - 1, this.XIndex).getNumber();
                    Number ynumber = dataBlock.get(fa - 1, this.YIndex).getNumber();
                    ppx = coordinateXAxis.transform(axisChart, xnumber).intValue();
                    ppy = coordinateYAxis.transform(axisChart, ynumber).intValue();
                } else {
                    ppx = px;
                    ppy = py;
                }
                Number xnumber1 = dataBlock.get(fa, this.XIndex).getNumber();
                Number ynumber1 = dataBlock.get(fa, this.YIndex).getNumber();
                px = coordinateXAxis.transform(axisChart, xnumber1).intValue();
                py = coordinateYAxis.transform(axisChart, ynumber1).intValue();
                previous = fa;

                int[] noutcomes = adjustToBorders(px, py, ppx, ppy, minX, maxX, minY, maxY, a, outcomes);
                if (noutcomes != null) {
                    px = outcomes[0];
                    py = outcomes[1];
                    ppx = outcomes[2];
                    ppy = outcomes[3];

                    this.shapes.get(shapeIndex).x1 = ppx;
                    this.shapes.get(shapeIndex).y1 = ppy;
                    this.shapes.get(shapeIndex).x2 = px;
                    this.shapes.get(shapeIndex).y2 = py;
                    gd.draw(this.shapes.get(shapeIndex));
                }
            }


        }
    }

    /* private int[] adjustToBorders(int px, int py, int ppx, int ppy, int minX, int maxX, int minY,
    int maxY, int[] a, int[] outcomes) {
    int[] ca;
    ca = checkCorners(px, py, ppx, ppy, px, py, minX, maxX, minY, maxY, a);
    if (ca != null) {
    px = ca[0];
    py = ca[1];
    }

    ca = checkCorners(ppx, ppy, ppx, ppy, px, py, minX, maxX, minY, maxY, a);
    if (ca != null) {
    ppx = ca[0];
    ppy = ca[1];
    }

    ca = checkX(px, py, ppx, ppy, px, py, minX, maxX, a);
    if (ca != null) {
    px = ca[0];
    py = ca[1];
    }


    ca = checkX(ppx, ppy, ppx, ppy, px, py, minX, maxX, a);
    if (ca != null) {
    ppx = ca[0];
    ppy = ca[1];
    }

    ca = checkY(px, py, ppx, ppy, px, py, minY, maxY, a);
    if (ca != null) {
    px = ca[0];
    py = ca[1];
    }

    ca = checkY(ppx, ppy, ppx, ppy, px, py, minY, maxY, a);
    if (ca != null) {
    ppx = ca[0];
    ppy = ca[1];
    }
    outcomes[0] = px;
    outcomes[1] = py;
    outcomes[2] = ppx;
    outcomes[3] = ppy;
    return outcomes;
    }*/
    private int[] adjustToBorders(int px, int py, int ppx, int ppy, int minX, int maxX, int minY,
            int maxY, int[] temp, int[] outcomes) {
        float a = (float) (ppy - py) / (float) (ppx - px);
        float b = ppy - (float) ppx * a;

        int npx = px;
        int npy = py;
        int nppx = ppx;
        int nppy = ppy;
        if ((npy > maxY && nppy > maxY) || (npy < minY && nppy < minY)) {
            return null;
        }

        if ((npx > maxX && nppx > maxX) || (npx < minX && nppx < minX)) {
            return null;
        }

        if (nppx == npx) {
            if (npy > maxY) {
                npy = maxY;
            } else if (npy < minY) {
                npy = minY;
            }

            if (nppy > maxY) {
                nppy = maxY;
            } else if (nppy < minY) {
                nppy = minY;
            }

            outcomes[0] = npx;
            outcomes[1] = npy;
            outcomes[2] = nppx;
            outcomes[3] = nppy;
            return outcomes;
        }


        if (nppy == npy) {

            if (npx > maxX) {
                npx = maxX;
            } else if (npx < minX) {
                npx = minX;
            }

            if (nppx > maxX) {
                nppx = maxX;
            } else if (nppx < minX) {
                nppx = minX;
            }

            outcomes[0] = npx;
            outcomes[1] = npy;
            outcomes[2] = nppx;
            outcomes[3] = nppy;
            return outcomes;
        }

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

    private int[] checkCorners(int cx, int cy, int ppx, int ppy, int px, int py, int minX, int maxX, int minY, int maxY, int[] a) {
        if (cx > maxX && cy < minY) {
            int lx = cx - maxX;
            int ly = minY - cy;
            if (lx <= ly) {
                int x = calculateXPoint(ppx, ppy, px, py, minY);
                cx = x;
                cy = minY;
            } else {
                int y = calculateYPoint(ppx, ppy, px, py, maxX);
                cx = maxX;
                cy = y;
            }

            a[0] = cx;
            a[1] = cy;
            return a;
        } else if (cx > maxX && cy > maxY) {
            int lx = cx - maxX;
            int ly = cy - maxY;
            if (lx <= ly) {
                int x = calculateXPoint(ppx, ppy, px, py, maxY);
                cx = x;
                cy = maxY;
            } else {
                int y = calculateYPoint(ppx, ppy, px, py, maxX);
                cx = maxX;
                cy = y;
            }
            a[0] = cx;
            a[1] = cy;
            return a;
        } else if (cx < minX && cy > maxY) {
            int lx = minX - cx;
            int ly = cy - maxY;
            if (lx <= ly) {
                int x = calculateXPoint(ppx, ppy, px, py, maxY);
                cx = x;
                cy = maxY;
            } else {
                int y = calculateYPoint(ppx, ppy, px, py, minX);
                cx = minX;
                cy = y;
            }
            a[0] = cx;
            a[1] = cy;
            return a;
        } else if (cx < minX && cy < minY) {
            int lx = minX - cx;
            int ly = minY - cy;
            if (lx <= ly) {
                int x = calculateXPoint(ppx, ppy, px, py, minY);
                cx = x;
                cy = minY;
            } else {
                int y = calculateYPoint(ppx, ppy, px, py, minX);
                cx = minX;
                cy = y;
            }
            a[0] = cx;
            a[1] = cy;
            return a;
        }

        return null;
    }

    private int[] checkX(int cx, int cy, int ppx, int ppy, int px, int py, int minX, int maxX, int[] a) {
        if (cx < minX) {
            if (ppy != py) {
                int y = calculateYPoint(ppx, ppy, px, py, minX);
                cy = y;
            }
            cx = minX;

            a[0] = cx;
            a[1] = cy;
            return a;
        } else if (cx > maxX) {
            if (ppy != py) {
                int y = calculateYPoint(ppx, ppy, px, py, maxX);
                cy = y;
            }
            cx = maxX;

            a[0] = cx;
            a[1] = cy;
            return a;
        }
        return null;
    }

    private int[] checkY(int cx, int cy, int ppx, int ppy, int px, int py, int minY, int maxY, int[] a) {
        if (cy < minY) {
            if (ppx != px) {
                int x = calculateXPoint(ppx, ppy, px, py, minY);
                cx = x;
            }
            cy = minY;
            a[0] = cx;
            a[1] = cy;
            return a;
        } else if (cy > maxY) {
            if (ppx != px) {
                int x = calculateXPoint(ppx, ppy, px, py, maxY);
                cx = x;
            }
            cy = maxY;
            a[0] = cx;
            a[1] = cy;
            return a;
        }
        return null;
    }

    private int calculateXPoint(int px, int py, int x, int y, int currentY) {
        float a = (float) (y - py) / (float) (x - px);
        float b = (float) y - a * (float) x;
        float o = currentY - b;
        o = o / a;
        return (int) o;
    }

    private int calculateYPoint(int px, int py, int x, int y, int currentX) {
        float a = (float) (py - y) / (x - px);
        float b = (float) y - a * (float) x;
        float o = a * (float) currentX + b;
        return (int) o;
    }

    @Override
    public Line2D.Double newShape() {
        return new Line2D.Double();
    }

    @Override
    public String getLabel() {
        return "Line";
    }
    private Color plotColor = Color.black;
    private BasicStroke plotStroke = new BasicStroke(1.f);
    private ConfigurationFrame configuraionFrame = null;

    @Override
    public JFrame getFrame() {
        return configuraionFrame;
    }

    public void setColor(Color color) {
        this.plotColor = color;
    }

    public void setStroke(BasicStroke stroke) {
        this.plotStroke = stroke;
    }

    @Override
    public byte[] save() {
        Writer writer = new Writer();
        try {
            writer.write(plotStroke.getLineWidth());
            writer.write(plotColor.getRed());
            writer.write(plotColor.getGreen());
            writer.write(plotColor.getBlue());
        } catch (IOException ex) {
            Logger.getLogger(PlotLineManager.class.getName()).log(Level.SEVERE, null, ex);
        }
        return writer.getBytes();
    }

    public String getDescription() {
        return "";
    }
}
