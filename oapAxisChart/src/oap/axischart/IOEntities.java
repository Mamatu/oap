package ogla.axischart;

import ogla.axischart.plugin.image.ImageComponent;
import ogla.axischart.plugin.image.ImageComponentBundle;
import ogla.axischart.lists.ListDataBundle;
import ogla.axischart.lists.ListDataRepository;
import ogla.axischart.lists.ListImageComponentBundle;
import ogla.axischart.lists.ListInstalledImageComponent;
import ogla.axischart.lists.ListListener;
import ogla.axischart.lists.ListsContainer;
import ogla.chart.IOEntity;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import ogla.core.util.Reader;
import ogla.core.util.Writer;
import ogla.core.data.DataBundle;
import ogla.core.data.DataRepository;

public class IOEntities {

    private ListsContainer listContainer;
    private ListDataBundle listDataBundle;
    private ListDataRepository listDataRepository;
    private ListImageComponentBundle listImageComponentBundle;
    private ListInstalledImageComponent listInstalledImageComponent;
    private List<RawData> rawsToRemove = new ArrayList<RawData>();

    public IOEntities(AxisChart chart, ListsContainer listContainer) {
        this.listContainer = listContainer;
        this.listDataBundle = listContainer.get(ListDataBundle.class);
        this.listDataRepository = listContainer.get(ListDataRepository.class);
        this.listImageComponentBundle = listContainer.get(ListImageComponentBundle.class);
        this.listInstalledImageComponent = listContainer.get(ListInstalledImageComponent.class);
        chart.addIOEntity(new DataRepositoryEntity(this.listDataRepository));
        chart.addIOEntity(new InstalledChartComponentEntity(listInstalledImageComponent));

        this.listDataBundle.addListListener(new ListListener<DataBundle>() {

            public void isAdded(DataBundle t) {
                for (int fa = 0; fa < dataRepositoryRaws.size(); fa++) {
                    RawData dataRepositoryRaw = dataRepositoryRaws.get(fa);
                    if (t.getSymbolicName().equals(dataRepositoryRaw.symbolicName)) {
                        rawsToRemove.add(dataRepositoryRaw);
                        t.loadToContainer(dataRepositoryRaw.bytes);
                    }
                }
                for (RawData raw : rawsToRemove) {
                    dataRepositoryRaws.remove(raw);
                }
                rawsToRemove.clear();
            }

            public void isRemoved(DataBundle t) {
            }
        });

        this.listImageComponentBundle.addListListener(new ListListener<ImageComponentBundle>() {

            public void isAdded(ImageComponentBundle t) {
                for (int fa = 0; fa < installedChartComponentRaws.size(); fa++) {
                    RawData raw = dataRepositoryRaws.get(fa);
                    if (t.getSymbolicName().equals(raw.symbolicName)) {
                        rawsToRemove.add(raw);
                        ImageComponent imageComponent = t.load(raw.bytes);
                        IOEntities.this.listInstalledImageComponent.add(imageComponent);
                    }
                }
                for (RawData raw : rawsToRemove) {
                    dataRepositoryRaws.remove(raw);
                }
                rawsToRemove.clear();
            }

            public void isRemoved(ImageComponentBundle t) {
            }
        });
    }
    private List<RawData> dataRepositoryRaws = new ArrayList<RawData>();

    private class RawData {

        public RawData(String symbolicName, byte[] bytes) {
            this.symbolicName = symbolicName;
            this.bytes = bytes;
        }
        public String symbolicName;
        public byte[] bytes;
    }

    private class DataRepositoryEntity implements IOEntity {

        private ListDataRepository listDataRepository = null;

        public DataRepositoryEntity(ListDataRepository listDataRepositoriesGroup) {
            this.listDataRepository = listDataRepositoriesGroup;
        }

        public void save(Writer writer) throws IOException {
            Writer localWriter = new Writer();
            int length = 0;
            for (int fa = 0; fa < listDataRepository.size(); fa++) {
                DataRepository dataRepository = listDataRepository.get(fa);
                byte[] bytes = dataRepository.save();
                if (bytes != null) {
                    length++;
                    String name = dataRepository.getBundle().getSymbolicName();
                    localWriter.write(name.length());
                    localWriter.write(name);
                    localWriter.write(bytes.length);
                    localWriter.write(bytes);
                }
            }
            writer.write(length);
            writer.write(localWriter.getBytes());
        }

        public void load(Reader reader) throws IOException, Reader.EndOfBufferException {
            int length = reader.readInt();
            for (int fa = 0; fa < length; fa++) {
                int length1 = reader.readInt();
                String name = reader.readString(length1);
                int length2 = reader.readInt();
                byte[] array = new byte[length2];
                reader.readBytes(array);
                dataRepositoryRaws.add(new RawData(name, array));
            }
        }
    }
    private List<RawData> installedChartComponentRaws = new ArrayList<RawData>();

    private class InstalledChartComponentEntity implements IOEntity {

        private ListInstalledImageComponent listInstalledImageComponent;

        public InstalledChartComponentEntity(ListInstalledImageComponent listInstalledImageComponent) {
            this.listInstalledImageComponent = listInstalledImageComponent;
        }

        public void save(Writer writer) throws IOException {
            Writer localWriter = new Writer();
            int length = 0;
            for (int fa = 0; fa < listInstalledImageComponent.size(); fa++) {
                ImageComponent imageComponent = listInstalledImageComponent.get(fa);
                byte[] bytes = imageComponent.save();
                if (bytes != null) {
                    String symbolicName = imageComponent.getBundle().getSymbolicName();
                    localWriter.write(symbolicName.length());
                    localWriter.write(symbolicName);
                    localWriter.write(bytes.length);
                    localWriter.write(bytes);
                    length++;
                }
            }
            writer.write(length);
            writer.write(localWriter.getBytes());
        }

        public void load(Reader reader) throws IOException, ogla.core.util.Reader.EndOfBufferException {
            int length = reader.readInt();
            for (int fa = 0; fa < length; fa++) {
                int length1 = reader.readInt();
                String name = reader.readString(length1);
                int length2 = reader.readInt();
                byte[] array = new byte[length2];
                reader.readBytes(array);
                installedChartComponentRaws.add(new RawData(name, array));
            }
        }
    }
}
