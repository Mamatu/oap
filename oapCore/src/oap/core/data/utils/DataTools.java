package ogla.core.data.utils;

import ogla.core.data.DataBlock;
import ogla.core.data.DataValue;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class DataTools {

    /**
     * DataRow interface to help manage data in repository.
     */
    public static interface DataRow {

        public int size();

        public DataValue get(int index);
    }

    /**
     * Default implementation of DataValue.
     */
    public static class DataValueImpl implements DataValue {

        private String label = "";
        private Number number;

        public DataValueImpl() {
        }

        public DataValueImpl(DataValue dataValue) {
            this.number = dataValue.getNumber();
            this.label = dataValue.getLabel();
        }

        public DataValueImpl(Number number) {
            this.number = number;
            this.label = String.valueOf(number);
        }

        public DataValueImpl(Number number, String label) {
            this.number = number;
            this.label = label;
        }

        public void setLabel(String label) {
            this.label = label;
        }

        public void setNumber(Number number) {
            this.number = number;
        }

        public String getLabel() {
            return label;
        }

        public Number getNumber() {
            return number;
        }
    }

    /**
     * Default implementation of DataRow.
     */
    public static class DataRowImpl implements DataTools.DataRow, Comparable<DataRowImpl> {

        private List<DataValueImpl> dataValues = new ArrayList<DataValueImpl>();
        protected int index = 0;

        public DataValueImpl getDataValueImpl(int index) {
            return dataValues.get(index);
        }

        public void addDataValue(DataValue dataValue) {
            this.dataValues.add(new DataValueImpl(dataValue));
        }

        public void removeDataValue(int index) {
            this.dataValues.remove(index);
        }

        public void removeDataValue(DataValue dataValue) {
            if (!(dataValue instanceof DataValueImpl)) {
                return;
            }
            DataValueImpl impl = (DataValueImpl) dataValue;
            boolean remove = this.dataValues.remove(impl);
        }

        public DataRowImpl() {
        }

        public DataRowImpl(DataValue dataValue, int repeat) {
            this(dataValue, repeat, 0);

        }

        public DataRowImpl(DataValue dataValue, int repeat, int index) {
            this.index = index;
            for (int fa = 0; fa < repeat; fa++) {
                this.dataValues.add(new DataValueImpl(dataValue));
            }
        }

        public DataRowImpl(List<DataValue> dataValues, int index) {
            this.index = index;
            for (DataValue dataValue : dataValues) {
                this.dataValues.add(new DataValueImpl(dataValue));
            }
        }

        public DataRowImpl(DataBlock dataBlock, int row) {
            this(dataBlock, row, 0);
        }

        public DataRowImpl(DataBlock dataBlock, int row, int index) {
            for (int fa = 0; fa < dataBlock.columns(); fa++) {
                this.dataValues.add(new DataValueImpl(dataBlock.get(row, fa)));
            }

            this.index = index;
        }

        public DataRowImpl(DataRow dataRow) {
            this(dataRow, 0);
        }

        public DataRowImpl(DataRow dataRow, int index) {
            for (int fa = 0; fa < dataRow.size(); fa++) {
                this.dataValues.add(new DataValueImpl(dataRow.get(fa)));
            }
            this.index = index;
        }

        public DataRowImpl(DataRowImpl dataRowImpl, int index) {
            this.index = index;
            this.dataValues.addAll(dataRowImpl.dataValues);
        }

        public DataRowImpl(DataRowImpl dataRowImpl) {
            this(dataRowImpl, 0);
        }

        public void add(Number number) {
            dataValues.add(new DataValueImpl(number));
        }

        public void add(Number number, String label) {
            dataValues.add(new DataValueImpl(number, label));
        }

        public void add(DataValue dataValue) {
            dataValues.add(new DataValueImpl(dataValue));
        }

        public DataValue get(int index) {
            return dataValues.get(index);
        }

        public int size() {
            return dataValues.size();
        }

        public int compareTo(DataRowImpl o) {
            double diff = this.get(index).getNumber().doubleValue() - o.get(index).getNumber().doubleValue();
            if (diff < 0) {
                return -1;
            } else if (diff > 0) {
                return 1;
            } else {
                return 0;
            }
        }
    }

    public static List<DataRowImpl> sortAsc(List<DataRowImpl> rows, int index) {
        List<DataTools.DataRowImpl> proxies = new ArrayList<DataTools.DataRowImpl>();
        for (DataRowImpl row : rows) {
            proxies.add(new DataRowImpl(row, index));
        }

        Collections.sort(proxies);

        rows.clear();

        for (DataRowImpl rowContainer : proxies) {
            rows.add(rowContainer);
        }
        return rows;
    }

    public static List<DataRowImpl> sortAsc(List<DataRowImpl> dataRows) {
        return DataTools.sortAsc(dataRows, 0);
    }

    public static class IntegerRef {

        public int i = 0;
    }

    public static DataValue[] add(DataValue dataValue, DataValue[] dataValues) {
        if (dataValues == null) {
            dataValues = new DataValue[1];
            dataValues[0] = dataValue;
            return dataValues;
        }

        DataValue[] ndataValues = new DataValue[dataValues.length + 1];
        System.arraycopy(ndataValues, 0, dataValues, 0, dataValues.length);
        ndataValues[dataValues.length] = dataValue;
        return ndataValues;
    }

    public static DataValue[] add(DataValue dataValue, DataValue[] dataValues, int end) {
        if (dataValues == null) {
            dataValues = new DataValue[end - 1];
            for (int fa = 0; fa < end - 1; fa++) {
                dataValues[fa] = dataValue;
            }
            return dataValues;
        }
        if (end <= dataValues.length) {
            return dataValues;
        }
        DataValue[] ndataValues = new DataValue[end];
        System.arraycopy(ndataValues, 0, dataValues, 0, dataValues.length);
        for (int fa = dataValues.length; fa < end; fa++) {
            ndataValues[fa] = dataValue;
        }
        return ndataValues;
    }
}
