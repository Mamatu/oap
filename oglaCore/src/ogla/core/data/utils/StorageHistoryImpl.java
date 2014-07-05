package ogla.core.data.utils;


import ogla.core.data.DataBundle;
import ogla.core.data.DataRepository;
import ogla.core.data.History;
import ogla.core.data.DataBlock;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel.MapMode;
import java.util.Collections;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import ogla.core.util.Reader;
import ogla.core.util.Writer;

/**
 * History which can write some numbers of repositories to file.
 * This class should be used only when set of repositories can occupy big part of memory.
 * @author marcin
 */
public class StorageHistoryImpl extends History {

    private String path = "data/tempspace/temp_repositories_groups_";
    private String pathTemp = "data/tempspace/temp_repositories_groups_temp_";
    private DataRepository dataRepositoriesGroup;

    /**
     * Return identificator which identify repositories to this history's group.
     * @return
     */
    public String getIdentificator() {
        return this.toString();
    }

    @Override
    public DataRepository getCurrent() {
        return dataRepositoriesGroup;
    }

    private class Element implements Comparable<Element> {

        public Element(DataRepository dataRepositoriesGroup, int index) {
            this.dataRepositoriesGroup = dataRepositoriesGroup;
            this.index = index;
        }

        public int compareTo(Element o) {
            if (this.counter == o.counter) {
                return 0;
            }
            if (this.counter > o.counter) {
                return 1;
            }
            return -1;

        }
        public DataRepository dataRepositoriesGroup;
        int index;
        public int counter = 0;
    }

    private Element getElement(int index) {
        for (int fa = 0; fa < elements.size(); fa++) {
            if (elements.get(fa).index == index) {
                return elements.get(fa);
            }
        }
        return null;
    }
    private List<Element> elements = new ArrayList<Element>();
    private int limit;
    private int numberForSaved;
    private int sizeInMemory = 0;
    private int numberOfUsing = 0;
    private DataBundle dataBundle;
    private File file = null;
    private File fileTemp = null;

    /**
     * 
     * @param limit - number of repositories in this history, after croosing this number by size of container,
     * saving mode should be started.
     * @param numberForSaved - how much repositories will be saved to file if limit was crossed.
     */
    public StorageHistoryImpl(DataBundle dataBundle, int limit, int numberForSaved, DataRepository dataRepositoriesGroup) {
        this.limit = limit;
        this.numberForSaved = numberForSaved;
        this.dataBundle = dataBundle;
        this.dataRepositoriesGroup = dataRepositoriesGroup;
        this.path = this.path + getIdentificator();
        this.pathTemp = this.pathTemp + getIdentificator();
        file = new File(path);
        fileTemp = new File(pathTemp);
    }


    /*
     * Remove files where history is stored.
     */
    public void removeFiles() {
        if (file.exists()) {
            file.delete();
        }
        if (fileTemp.exists()) {
            fileTemp.delete();
        }
    }

    private void saveImpl() {
        Collections.sort(elements);
        List<Element> savedElements = new ArrayList<Element>();
        int counter = 0;
        while (counter < elements.size() && savedElements.size() < numberForSaved) {
            Element element = elements.get(counter);
            if (element.dataRepositoriesGroup != null) {
                savedElements.add(element);

            }
            counter++;
        }
        Element[] savedElementsArray = new Element[savedElements.size()];
        savedElementsArray = savedElements.toArray(savedElementsArray);
        int size = this.save(savedElementsArray);
        sizeInMemory -= size;
        for (Element countingElement : savedElements) {
            countingElement.dataRepositoriesGroup = null;
        }
    }

    /**
     * Add group into container.
     * @param dataRepositoryGroup
     */
    public void add(DataRepository dataRepositoryGroup) {
        elements.add(new Element(dataRepositoryGroup, elements.size()));
        sizeInMemory++;
        if (sizeInMemory > limit) {
            saveImpl();
        }
    }

    @Override
    public int size() {
        return elements.size();
    }

    @Override
    public DataRepository get(int index) {
        Element countingElement = getElement(index);
        countingElement.counter++;
        numberOfUsing++;
        if (countingElement.dataRepositoriesGroup != null) {
            return countingElement.dataRepositoriesGroup;
        }
        int[] indexes = new int[1];
        indexes[0] = index;
        Map<Integer, DataRepository> map = this.load(dataBundle, indexes);
        indexes = null;
        DataRepository dataRepositoriesGroup = map.get((Integer) index);
        countingElement.dataRepositoriesGroup = dataRepositoriesGroup;
        sizeInMemory++;
        return dataRepositoriesGroup;
    }

    private int save(Element[] elements) {
        FileOutputStream fileOutputStream = null;
        int numberOfSaved = 0;
        try {
            if (!file.exists()) {
                file.createNewFile();
            }
            fileOutputStream = new FileOutputStream(file, true);
            Writer localWriter = new Writer();
            for (Element element : elements) {
                localWriter.write(element.index);
                byte[] bytes = element.dataRepositoriesGroup.save();
                localWriter.write(bytes.length);
                localWriter.write(bytes);
                numberOfSaved++;
            }
            Writer writer = new Writer();
            final int totalSize = localWriter.size() + 4;
            writer.write(totalSize);
            writer.write(elements.length);
            writer.write(localWriter.getBytes());
            fileOutputStream.write(writer.getBytes());
        } catch (FileNotFoundException ex) {
            Logger.getLogger(StorageHistoryImpl.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException exception) {
            Logger.getLogger(StorageHistoryImpl.class.getName()).log(Level.SEVERE, null, exception);
        } finally {
            try {
                fileOutputStream.close();
            } catch (IOException ex) {
                Logger.getLogger(StorageHistoryImpl.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        return numberOfSaved;
    }
    private byte[] word = new byte[4];

    private boolean isIndex(int index, List<Integer> numbers) {
        for (int fa = 0; fa < numbers.size(); fa++) {
            if (numbers.get(fa) == index) {
                return true;
            }
        }
        return false;
    }

    private Map<Integer, DataRepository> load(DataBundle bundleData, int[] indexes) {
        FileInputStream fileInputStream = null;
        Map<Integer, DataRepository> map = new HashMap<Integer, DataRepository>();
        FileOutputStream fileOutputStream = null;
        try {
            List<Integer> indecies = new ArrayList<Integer>();
            for (int i : indexes) {
                indecies.add(i);
            }
            Reader reader = new Reader();
            fileInputStream = new FileInputStream(file);
            Writer writer = new Writer();
            while (fileInputStream.read(word) != -1) {
                int sizeOfBlock = reader.setBytes(word).readInt();
                MappedByteBuffer mappedByteBuffer =
                        fileInputStream.getChannel().map(MapMode.READ_ONLY, fileInputStream.getChannel().position(), sizeOfBlock);
                int numberOfRepositories = mappedByteBuffer.getInt();
                int counter = numberOfRepositories;
                Writer localWriter = new Writer();
                for (int fa = 0; fa < numberOfRepositories; fa++) {
                    int index = mappedByteBuffer.getInt();
                    int sizeOfRepository = mappedByteBuffer.getInt();
                    byte[] content = new byte[sizeOfRepository];
                    mappedByteBuffer.get(content);
                    if (isIndex(index, indecies)) {
                        indecies.remove((Integer) index);
                        DataRepository dataRepositoriesGroup = bundleData.load(content);
                        map.put(index, dataRepositoriesGroup);
                        counter--;
                    } else {
                        localWriter.write(index);
                        localWriter.write(sizeOfRepository);
                        localWriter.write(content);
                    }
                }
                fileInputStream.skip(sizeOfBlock);
                byte[] bytesOfLocalWriter = localWriter.getBytes();
                if (bytesOfLocalWriter.length != 0) {
                    writer.write(bytesOfLocalWriter.length + 4);
                    writer.write(counter);
                    writer.write(bytesOfLocalWriter);
                }
            }
            fileOutputStream = new FileOutputStream(fileTemp);
            fileOutputStream.write(writer.getBytes());
            fileOutputStream.close();
            fileOutputStream = null;
            fileInputStream.close();
            fileInputStream = null;
            file.delete();
            fileTemp.renameTo(file);

        } catch (Reader.EndOfBufferException ex) {
            Logger.getLogger(StorageHistoryImpl.class.getName()).log(Level.SEVERE, null, ex);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(StorageHistoryImpl.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(StorageHistoryImpl.class.getName()).log(Level.SEVERE, null, ex);
        } finally {
            try {
                if (fileInputStream != null) {
                    fileInputStream.close();
                }
                if (fileOutputStream != null) {
                    fileOutputStream.close();
                }
            } catch (IOException ex) {
                Logger.getLogger(StorageHistoryImpl.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        return map;
    }
}
