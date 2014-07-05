package ogla.grid.util;

import ogla.core.data.DataValue;
import ogla.axischart2D.AxisChart2D;
import ogla.axischart2D.Ticks;
import ogla.axischart2D.coordinates.CoordinateTransformer;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.Stroke;
import java.awt.geom.Line2D;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author marcin
 */
public abstract class GridManager {

    protected Font font;
    protected List<Line2D.Double> lines = new ArrayList<Line2D.Double>(0);
    protected AxisChart2D axisChart;
    protected LocationManager locationManager;
    protected int expectedLength = 10;
    private CoordinateTransformer coordinateTransformer;
    private boolean isDisplayedText = true;

    public void displayText(boolean b) {
        isDisplayedText = b;
    }

    public AxisChart2D getAxisChart() {
        return axisChart;
    }

    public GridManager(AxisChart2D axisChart, Ticks ticks, CoordinateTransformer coordinateTransformer) {
        this.axisChart = axisChart;
        locationManager = new LocationManager(axisChart, ticks, coordinateTransformer);
        this.coordinateTransformer = coordinateTransformer;
    }

    public int fontSize() {
        return font.getSize();
    }

    public int getLength() {
        return expectedLength;
    }

    public void setLength(int length) {
        this.expectedLength = length;
    }

    private final void adjustContainerSize(int length) {
        if (length > lines.size()) {
            int size = length;
            for (int fa = lines.size(); fa < size; fa++) {
                lines.add(new Line2D.Double());
            }
        } else if (length < lines.size()) {
            int size = lines.size();
            for (int fa = length; fa < size; fa++) {
                lines.remove(lines.size() - 1);
            }
        }
    }

    public abstract Number getPosition(int index);
    private Stroke defaultStroke = null;
    private Color defaultColor = null;

    private void initializeAWTAttributes(Graphics2D gd) {
        defaultColor = gd.getColor();
        gd.setColor(this.getColor());
        defaultStroke = gd.getStroke();
        if (getStroke(gd) != null) {
            gd.setStroke(getStroke(gd));
        }
    }

    private void destroyAWTAttributes(Graphics2D gd) {
        gd.setColor(defaultColor);
        gd.setStroke(defaultStroke);
    }
    protected BasicStroke basicStroke = null;

    public void setBasicStroke(BasicStroke basicStroke) {
        this.basicStroke = basicStroke;
    }

    public abstract void drawLine(Graphics2D gd, Line2D.Double line, int position);

    public abstract void drawTick(Graphics2D gd, DataValue tick, int position);
    protected Color color = new Color(0.5f, 0.5f, 0.5f);

    public Color getColor() {
        return new Color(0.8f, 0.8f, 0.8f);
    }

    public void setColor(Color color) {
        this.color = new Color(color.getRed(), color.getGreen(), color.getBlue(), color.getAlpha());
    }

    protected abstract Stroke getStroke(Graphics2D gd);

    protected abstract boolean overlapCondition(DataValue tick1, Number pos1,
            DataValue tick2, Number pos2, List<DataValue> values, List<Number> positions);
    private List<DataValue> dataValues = new ArrayList<DataValue>();
    private List<Number> positions = new ArrayList<Number>();

    private class Tuple {

        public Number position;
        public DataValue value;

        public Tuple(DataValue value, Number position) {
            this.value = value;
            this.position = position;
            positions.add(position);
            dataValues.add(value);
        }
    }

    private boolean overlap(Tuple tuple, Tuple[] tuples) {
        for (Tuple comTuple : tuples) {
            if (comTuple == null) {
                return false;
            }
            if (overlapCondition(comTuple.value, comTuple.position, tuple.value, tuple.position, dataValues, positions)) {
                return true;
            }
        }
        return false;
    }

    public void manage(Graphics2D gd) {
        updateFont(gd);
        int trueLength = 0;
        Tuple[] tuples = new Tuple[expectedLength];
        for (int fa = 0; fa
                < expectedLength; fa++) {
            Number position = getPosition(fa);
            DataValue valueData = locationManager.nearestTick(position.intValue());
            if (coordinateTransformer.inScope(axisChart, valueData.getNumber().floatValue())) {
                Number localPosition = coordinateTransformer.transform(axisChart, valueData.getNumber().floatValue());
                Tuple tuple = new Tuple(valueData, localPosition);
                if (!overlap(tuple, tuples)) {
                    tuples[trueLength] = tuple;
                    trueLength++;
                    if (isDisplayedText) {
                        drawTick(gd, valueData, localPosition.intValue());
                    }
                }
            }
        }
        adjustContainerSize(trueLength);
        initializeAWTAttributes(gd);
        for (int fa = 0; fa < trueLength; fa++) {
            Number position = tuples[fa].position;
            drawLine(gd, lines.get(fa), position.intValue());
        }
        destroyAWTAttributes(gd);
        tuples = null;
    }

    public void updateFont(Graphics2D gd) {
        Font newfont = gd.getFont();
        if (font != gd.getFont()) {
            font = newfont;
        }
    }

    public abstract class Border {

        public abstract void drawTick(Graphics2D gd, DataValue tick, float x, float y);
    }
}
