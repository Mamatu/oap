package ogla.datafitting;

import ogla.excore.util.Writer;
import ogla.exdata.listener.RepositoryListener;
import ogla.exdata.DataBundle;
import ogla.exdata.DataRepository;
import ogla.exdata.History;
import ogla.exdata.DataBlock;
import ogla.exdata.DataValue;
import ogla.exdata.util.DataTools;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public abstract class Operation implements Serializable {

    protected DataBundle bundleData;

    public Operation(DataBundle bundleData) {
        this.bundleData = bundleData;
    }

    public static class DataRowImpl implements DataTools.DataRow, Comparable<DataRowImpl> {

        public DataValue[] values;

        public DataRowImpl(int al) {
            values = new DataValue[al];
        }

        public DataRowImpl(float x, float y) {
            values = new DataValue[2];
            values[0] = new ValueDataImpl(x);
            values[1] = new ValueDataImpl(y);
        }

        public DataRowImpl(float x, String label1, float y, String label2) {
            values = new DataValue[2];
            ValueDataImpl value1 = new ValueDataImpl(x, label1);
            ValueDataImpl value2 = new ValueDataImpl(y, label2);
            values[0] = value1;
            values[1] = value2;
        }

        public DataValue get(int index) {
            return values[index];
        }

        public int size() {
            return values.length;
        }

        public int compareTo(DataRowImpl o) {
            if (this.values[0].getNumber().floatValue() > o.values[0].getNumber().floatValue()) {
                return 1;
            } else if (this.values[0].getNumber().floatValue() < o.values[0].getNumber().floatValue()) {
                return -1;
            }
            return 0;
        }

        public static class ValueDataImpl implements DataValue {

            Float f;
            String label;

            public ValueDataImpl(float v) {
                this.f = v;
                this.label = String.valueOf(v);
            }

            public ValueDataImpl(float v, String label) {
                this.f = v;
                this.label = label;
            }

            public String getLabel() {
                return label;
            }

            public Number getNumber() {
                return f;
            }
        }
    }

    public static class DataBlockImpl implements DataBlock {

        public DataValue get(int row, int column) {
            return rows.get(row).values[column];
        }

        public int rows() {
            return rows.size();
        }

        public int columns() {
            if (rows.size() == 0) {
                return 0;
            }
            return rows.get(rows.size() - 1).size();
        }
        public List<DataRowImpl> rows = new ArrayList<DataRowImpl>();
        private DataRepositoryImpl dataRepository;

        public DataBlockImpl(DataRepositoryImpl dataRepository) {
            this.dataRepository = dataRepository;
        }

        public void addRepositoryListener(RepositoryListener listener) {
        }

        public void removeRepositoryListener(RepositoryListener listener) {
        }

        public DataRepositoryImpl getDataRepository() {
            return dataRepository;
        }
    }

    public static class DataRepositoryImpl implements DataRepository {

        protected DataBundle dataBundle;

        public DataRepositoryImpl(DataBundle dataBundle) {
            this.dataBundle = dataBundle;
        }
        List<DataBlock> dataBlocks = new ArrayList<DataBlock>();

        @Override
        public DataBlock get(int index) {
            return dataBlocks.get(index);
        }

        @Override
        public int size() {
            return dataBlocks.size();
        }

        @Override
        public DataBundle getBundle() {
            return dataBundle;
        }

        @Override
        public History getHistory() {
            return null;
        }

        @Override
        public byte[] save() {
            Writer writer = new Writer();
            try {
                writer.write(0);
                writer.write(this.getLabel().length());
                writer.write(this.getLabel());
                writer.write(this.getDescription().length());
                writer.write(this.getDescription());

                writer.write(dataBlocks.size());
                for (int x = 0; x < dataBlocks.size(); x++) {
                    DataBlockImpl dataRepositoryImpl = (DataBlockImpl) dataBlocks.get(x);
                    writer.write(dataRepositoryImpl.rows.size());
                    for (int fa = 0; fa < dataRepositoryImpl.rows.size(); fa++) {
                        DataRowImpl lineDataImpl = dataRepositoryImpl.rows.get(fa);
                        writer.write(lineDataImpl.values.length);
                        for (DataValue valueDataImpl : lineDataImpl.values) {
                            writer.write(valueDataImpl.getNumber().floatValue());
                            writer.write(valueDataImpl.getLabel().length());
                            writer.write(valueDataImpl.getLabel());
                        }
                    }
                }
            } catch (IOException ex) {
                Logger.getLogger(DataBlockImpl.class.getName()).log(Level.SEVERE, null, ex);
            }
            return writer.getBytes();
        }
        public String label = "";
        public String description = "";

        public void setLabel(String label) {
            this.label = label;
        }

        @Override
        public String getLabel() {
            return label;
        }

        public void setDescription(String d) {
            this.description = d;
        }

        @Override
        public String getDescription() {
            return description;
        }

        public void addRepositoryListener(RepositoryListener listener) {
        }

        public void removeRepositoryListener(RepositoryListener listener) {
        }
    }

    public abstract DataBlock modify(DataBlock dataBlock, DataRepositoryImpl dataRepository,
            float begin, float end, float step, float degree, int x, int y);

    public abstract String getLabel();

    @Override
    public String toString() {
        return this.getLabel();

    }

    public abstract String getSuffix();
}
