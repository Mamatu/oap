package ogla.core.gui;

import ogla.core.util.Reader;
import ogla.core.util.Writer;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import ogla.core.data.DataBundle;
import ogla.core.data.DataRepository;

public class RepositoriesIO {

    private String pathToVirtualSpace = "data/virtualspace/vs_repositories";

    protected class RepositoryInfo {

        public String symbolicName = "";
        public String label = "";
        public byte[] content = null;
        public boolean wasInstalled = false;

        @Override
        public String toString() {
            return label + " (" + symbolicName + ")";
        }
    }

    protected class BundleInfo {

        public String symbolicName = "";
        public byte[] content = null;
    }
    private List<RepositoryInfo> repositoryInfos = new ArrayList<RepositoryInfo>();
    private List<BundleInfo> bundleInfos = new ArrayList<BundleInfo>();
    private List<DataBundle> bundles = new ArrayList<DataBundle>();

    protected synchronized List<RepositoryInfo> getInfos() {
        return repositoryInfos;
    }

    private void nload(byte[] all) {
        try {
            Reader reader = new Reader(all);
            int numberOfBundles = reader.readInt();
            for (int x = 0; x < numberOfBundles; x++) {
                int length = reader.readInt();
                String symbolicName = reader.readString(length);
                length = reader.readInt();
                byte[] bundleInfo = new byte[length];
                reader.readBytes(bundleInfo);
                BundleInfo bundleInfo1 = new BundleInfo();
                bundleInfo1.content = bundleInfo;
                bundleInfo1.symbolicName = symbolicName;
                bundleInfos.add(bundleInfo1);
                int numberOfRepositories = reader.readInt();
                for (int fa = 0; fa < numberOfRepositories; fa++) {
                    length = reader.readInt();
                    String label = "no_label";
                    if (length > 0) {
                        label = reader.readString(length);
                    }
                    length = reader.readInt();
                    byte[] bytesRepresentation = new byte[length];
                    reader.readBytes(bytesRepresentation);


                    RepositoryInfo info = new RepositoryInfo();
                    info.content = bytesRepresentation;
                    info.label = label;
                    info.symbolicName = symbolicName;
                    info.wasInstalled = false;
                    repositoryInfos.add(info);
                }
            }
            for (DataBundle bundleData : bundles) {
                for (RepositoryInfo info : repositoryInfos) {
                    tryInstall(info, bundleData);
                }
            }

        } catch (Reader.EndOfBufferException ex) {
            System.err.print(ex.getMessage());
            Logger.getLogger(RepositoriesIO.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(RepositoriesIO.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public void load() {
        try {
            File file = new File(pathToVirtualSpace);
            if (!file.exists()) {
                return;
            }
            RandomAccessFile randomAccessFile = new RandomAccessFile(file, "rw");
            byte[] bsize = new byte[4];
            randomAccessFile.read(bsize);
            Reader reader = new Reader(bsize);
            int size = reader.readInt();
            byte[] all = new byte[size];
            randomAccessFile.read(all);
            randomAccessFile.close();
            if (size != 0) {
                nload(all);
            }
        } catch (Reader.EndOfBufferException ex) {
            System.err.print(ex.getMessage());
            Logger.getLogger(RepositoriesIO.class.getName()).log(Level.SEVERE, null, ex);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(RepositoriesIO.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(RepositoriesIO.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public void save() {
        try {
            File file = new File(pathToVirtualSpace);
            file.delete();
            file.createNewFile();
            int counter = 0;
            Writer writer = new Writer();
            for (DataBundle dataBundle : bundles) {
                if (dataBundle.canBeSaved()) {
                    String symbolicName = dataBundle.getSymbolicName();
                    writer.write(symbolicName.length());
                    writer.write(symbolicName);
                    byte[] info = dataBundle.saveDataBundle();
                    if (info != null) {
                        writer.write(info.length);
                        writer.write(info);
                    } else {
                        writer.write(0);
                    }
                    counter++;
                    writer.write(dataBundle.size());
                    for (int fa = 0; fa < dataBundle.size(); fa++) {
                        DataRepository dataRepository = dataBundle.get(fa);
                        if (dataRepository.getLabel() == null || dataRepository.getLabel().length() == 0) {
                            writer.write(0);
                        } else {
                            writer.write(dataRepository.getLabel().length());
                            writer.write(dataRepository.getLabel());
                        }
                        byte[] bytes = dataRepository.save();
                        if (bytes != null) {
                            writer.write(bytes.length);
                            writer.write(bytes);
                        } else {
                            writer.write(0);
                        }
                    }
                }
            }
            RandomAccessFile randomAccessFile = new RandomAccessFile(file, "rw");

            byte[] bytes = writer.getBytes();
            byte[] size = new byte[4];
            byte[] bcounter = new byte[4];
            writer.fromInt(bytes.length + 4, size);
            writer.fromInt(counter, bcounter);
            byte[] out = new byte[8 + bytes.length];
            System.arraycopy(size, 0, out, 0, 4);
            System.arraycopy(bcounter, 0, out, 4, 4);
            System.arraycopy(bytes, 0, out, 8, bytes.length);
            randomAccessFile.write(out);
            randomAccessFile.close();
        } catch (IOException ex) {
            Logger.getLogger(RepositoriesIO.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void tryLoadInfo(BundleInfo info, DataBundle dataBundle) {
        if (dataBundle.canBeSaved() && info.symbolicName.equals(dataBundle.getSymbolicName())) {
            dataBundle.loadBundleData(info.content);
        }
    }

    private void tryInstall(RepositoryInfo info, DataBundle dataBundle) {
        if (dataBundle.canBeSaved() && !info.wasInstalled && info.symbolicName.equals(dataBundle.getSymbolicName())) {
            dataBundle.loadToContainer(info.content);
        }
    }

    private void tryUninstall(BundleInfo info, DataBundle bundleData) {
        if (info.symbolicName.equals(bundleData.getSymbolicName())) {
            bundleInfos.remove(info);
        }
    }

    private void tryUninstall(RepositoryInfo info, DataBundle bundleData) {
        if (info.symbolicName.equals(bundleData.getSymbolicName())) {
            repositoryInfos.remove(info);
        }
    }

    public void add(DataBundle bundleData) {
        if (!bundles.contains(bundleData) && bundleData.canBeSaved()) {
            bundles.add(bundleData);
            for (BundleInfo info : bundleInfos) {
                tryLoadInfo(info, bundleData);
            }
            for (RepositoryInfo info : repositoryInfos) {
                tryInstall(info, bundleData);
            }
        }
    }

    public void remove(DataBundle bundleData) {
        if (bundles.contains(bundleData)) {
            bundles.remove(bundleData);
            for (BundleInfo info : bundleInfos) {
                tryUninstall(info, bundleData);
            }
            for (RepositoryInfo info : repositoryInfos) {
                tryUninstall(info, bundleData);
            }
        }
    }
}
