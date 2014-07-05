package ogla.axischart2D;

import ogla.core.data.DataValue;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author marcin
 */
public class Ticks {

    protected Number lowTick = new Float(0.f);
    protected Number greatTick = new Float(10.f);
    protected float range = 10.f;
    protected List<DataValue> displayedTicks;
    protected int labelsLength = 4;
    protected int sizeOfList = 10;
    protected float step = 1.f;

    public Ticks() {
    }

    public void setStep(float step) {
        this.step = step;
    }

    public float getStep() {
        return step;
    }

    public void setLowestTick(Number lowTick) {
        this.lowTick = lowTick;
    }

    public void setGreatestTick(Number greatTick) {
        this.greatTick = greatTick;
    }

    public Number getLowestTick() {
        return lowTick;
    }

    public Number getGreatestTick() {
        return greatTick;
    }

    class ValueDataImpl implements DataValue {

        protected float f;
        protected String label;

        public ValueDataImpl(float f) {
            this.f = f;
            label = String.valueOf(f);
        }

        public void setLabel(String label) {
            this.label = label;
        }

        public void setNumber(float f) {
            this.f = f;
        }

        public String getLabel() {
            Double d = null;
            try {
                d = Double.parseDouble(label);
            } catch (NumberFormatException nfe) {
                d = null;
            }
            if (d == null) {
                if (label.length() > labelsLength) {
                    String sub = label.substring(0, labelsLength);
                    sub = sub + "...";
                    return sub;
                }
                return label;
            }
            if (label.length() < labelsLength) {
                return label;
            }
            int i = label.indexOf('.');
            if (labelsLength - 1 < i) {
                String sub = label.substring(0, labelsLength);
                sub = sub + "...";
                return sub;

            } else if (labelsLength - 1 == i) {
                String sub = label.substring(0, labelsLength);
                sub = sub + "0";
                return sub;
            } else {
                String sub = label.substring(0, labelsLength);
                return sub;
            }
        }

        public Number getNumber() {
            return f;
        }
    }

    private void set(float[] displayedTicks) {
        for (final float f : displayedTicks) {
            this.displayedTicks.add(new ValueDataImpl(f));
        }
    }

    private void set(float[] displayedTicks, int begin, int end) {
        for (int fa = begin; fa < end; fa++) {
            this.displayedTicks.add(new ValueDataImpl(displayedTicks[fa]));
        }
    }

    private void fill(float[] displayedTicks, int begin, int end) {
        for (int fa = begin; fa < end; fa++) {
            ValueDataImpl impl = (ValueDataImpl) this.displayedTicks.get(fa);
            impl.setNumber(displayedTicks[fa]);
            impl.setLabel(String.valueOf(displayedTicks[fa]));
        }
    }

    public void setDisplayedTicks(List<Float> list) {
        float[] ticks = new float[list.size()];
        for (int fa = 0; fa < list.size(); fa++) {
            ticks[fa] = list.get(fa);
        }
        setDisplayedTicks(ticks);
    }
    private List<Float> floats = new ArrayList<Float>();

    public void setDisplayedTicks(float min, float max, float step) {
        float current = min;
        int index = 0;
        while (current <= max) {
            if (index < floats.size()) {
                floats.set(index, current);
            } else {
                floats.add(current);
            }
            current += step;
            index++;
        }
        setDisplayedTicks(floats);
    }

    public void setDisplayedTicks(float[] displayedTicks) {
        if (this.displayedTicks == null) {
            this.displayedTicks = new ArrayList<DataValue>();
            set(displayedTicks);
            sizeOfList = this.displayedTicks.size();
        } else {
            if (this.displayedTicks.size() < displayedTicks.length) {
                fill(displayedTicks, 0, this.displayedTicks.size());
                set(displayedTicks, this.displayedTicks.size(), displayedTicks.length);
                this.sizeOfList = displayedTicks.length;
            } else {
                fill(displayedTicks, 0, displayedTicks.length);
                this.sizeOfList = displayedTicks.length;
            }
        }
        float min = this.getDataValue(0).getNumber().floatValue();
        float max = this.getDataValue(this.displayedTicks.size() - 1).getNumber().floatValue();
        this.range = max - min;
        if (this.range < 0) {
            this.range = -this.range;
        }
    }

    public float getRange() {
        return range;
    }

    public void setRange(float range) {
        float min = this.getDataValue(0).getNumber().floatValue();
        float max = this.getDataValue(this.displayedTicks.size() - 1).getNumber().floatValue();
        float computeRange = max - min;
        if (range < computeRange) {
            this.range = range;
        }
    }

    public DataValue getDataValue(int index) {
        return displayedTicks.get(index);
    }

    public int getSizeOfList() {
        return sizeOfList;
    }

    public void setLengthOfLabels(int length) {
        this.labelsLength = length;
    }

    public int getLengthOfLabels() {
        return labelsLength;
    }
    private float dynamicStep = 1.f;

    public void setDynamicStep(float step) {
        this.dynamicStep = step;
    }

    public float getDynamicStep() {
        return dynamicStep;
    }
}
