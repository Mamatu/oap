package ogla.datacsv;

import ogla.core.data.DataValue;

public class DataValueImpl implements DataValue {

    public String label;
    public Number number;

    public DataValueImpl(Number number, String label) {
        this.label = label;
        this.number = number;
    }

    public String getLabel() {
        return label;
    }

    public Number getNumber() {
        return number;
    }
}
