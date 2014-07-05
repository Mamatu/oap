package ogla.datacsv;

import ogla.core.util.Writer;
import ogla.core.data.DataBundle;
import ogla.core.data.DataRepository;
import ogla.core.data.DataBlock;
import ogla.core.data.DataValue;
import ogla.core.data.History;
import ogla.core.data.listeners.RepositoryListener;
import ogla.core.data.utils.DataTools;
import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

public class DataRepositoryImpl implements DataRepository {

    public String label;
    public String absolutePath;
    public DataBlockImpl dataBlockImpl = new DataBlockImpl(this);
    public DataBundle dataBundle;

    public DataRepositoryImpl(File file, DataBundle dataBundle) {
        this.label = file.getName();
        this.absolutePath = file.getAbsolutePath();
        this.dataBundle = dataBundle;
    }

    public DataRepositoryImpl(DataBundle dataBundle) {
        this.dataBundle = dataBundle;
    }

    public void set(String label, String absolutePath) {
        this.label = label;
        this.absolutePath = absolutePath;
    }

    public void setFile(File file) {
        this.absolutePath = file.getAbsolutePath();
        this.label = file.getName();
    }

    @Override
    public byte[] save() {
        Writer writer = new Writer();
        writer.write(0);
        writer.write(this.getLabel().length());
        writer.write(this.getLabel());
        writer.write(this.absolutePath.length());
        writer.write(this.absolutePath);
        writer.write(dataBlockImpl.rows.size());
        for (int fa = 0; fa < dataBlockImpl.rows.size(); fa++) {
            DataTools.DataRowImpl row = dataBlockImpl.rows.get(fa);
            writer.write(row.size());
            for (int fc = 0; fc < row.size(); fc++) {
                DataValue dataValue = row.get(fc);
                writer.write(dataValue.getLabel().length());
                writer.write(dataValue.getLabel());
                
                writer.write(dataValue.getNumber().doubleValue());
            }
        }
        return writer.getBytes();
    }

    @Override
    public DataBlock get(int index) {
        return dataBlockImpl;
    }

    @Override
    public int size() {
        return 1;
    }

    @Override
    public DataBundle getBundle() {
        return dataBundle;
    }

    @Override
    public History getHistory() {
        return null;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    @Override
    public String getLabel() {
        return label;
    }

    @Override
    public String getDescription() {
        return "";
    }

    @Override
    public String toString() {
        return label;
    }

    public void addRepositoryListener(RepositoryListener listener) {
        this.dataBlockImpl.addRepositoryListener(listener);
    }

    public void removeRepositoryListener(RepositoryListener listener) {
        this.dataBlockImpl.removeRepositoryListener(listener);
    }
}
