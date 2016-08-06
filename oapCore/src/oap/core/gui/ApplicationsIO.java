package ogla.core.gui;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import ogla.core.application.Application;
import ogla.core.application.ApplicationBundle;
import ogla.core.util.GenericArray;
import ogla.core.util.Reader;
import ogla.core.util.Reader.EndOfBufferException;
import ogla.core.util.Writer;

public class ApplicationsIO {

    private class SavedNamesArray implements GenericArray<String> {

        public List<String> list = new ArrayList<String>();

        public String get(int index) {
            return list.get(index);
        }

        public int size() {
            return list.size();
        }

        public String id() {
            return "saved_names_array";
        }
    }

    public GenericArray getGenericArray() {
        return namesArray;
    }
    private SavedNamesArray namesArray = new SavedNamesArray();
    private DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
    private String path = "";

    public ApplicationsIO(String path) {
        this.path = path;
        String[] info = getInfoFromFile();
        if (info != null && info.length > 0) {
            for (int fa = 1; fa < info.length; fa += 3) {
                namesArray.list.add(info[fa]);
            }
        }

    }

    private class ApplicationObj {

        public Application application;
        public List<ApplicationDateObj> applicationDateObjs = new ArrayList<ApplicationDateObj>();
        public String nameOfApplication;
        public String symbolicNameOfBundle;

        public ApplicationObj(Application application, String nameOfApplication, String symbolicNameOfBundle) {
            this.application = application;
            this.nameOfApplication = nameOfApplication;
            this.symbolicNameOfBundle = symbolicNameOfBundle;
        }

        @Override
        public String toString() {
            return nameOfApplication;
        }
    }

    private class ApplicationDateObj {

        public byte[] representation;
        public String dateOfSaving;
        public int id;

        public ApplicationDateObj(byte[] byteRepresentation) {
            this.representation = byteRepresentation;
            this.dateOfSaving = dateFormat.format(Calendar.getInstance().getTime());
        }

        @Override
        public String toString() {
            return dateOfSaving;
        }
    }
    private Map<String, ApplicationObj> labels = new HashMap<String, ApplicationObj>();
    private List<String> infos = new ArrayList<String>();

    String[] getInfoFromFile() {
        RandomAccessFile raf = null;
        File file = null;
        try {
            file = new File(path);
            if (!file.exists()) {
                return null;
            }
            raf = new RandomAccessFile(file, "r");
            while (raf.getFilePointer() < raf.length()) {
                int length = readInt(raf);
                infos.add(readString(raf, length));
                length = readInt(raf);
                infos.add(readString(raf, length));
                length = readInt(raf);
                infos.add(readString(raf, length));
                length = readInt(raf);
                raf.skipBytes(length);
            }

        } catch (EndOfBufferException ex) {
            Logger.getLogger(ApplicationsIO.class.getName()).log(Level.SEVERE, null, ex);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(ApplicationsIO.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(ApplicationsIO.class.getName()).log(Level.SEVERE, null, ex);
        } finally {
            if (raf != null) {
                try {
                    raf.close();
                } catch (IOException ex) {
                    Logger.getLogger(ApplicationsIO.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        }
        String[] out = new String[infos.size()];
        out = infos.toArray(out);
        infos.clear();
        return out;
    }

    public String addApplication(Application application, ApplicationBundle bundleApplication, String label) {
        if (!labels.containsKey(label)) {
            ApplicationObj applicationObj = new ApplicationObj(application, label, bundleApplication.getSymbolicName());
            byte[] representation = application.save();
            ApplicationDateObj applicationDateObj = new ApplicationDateObj(representation);
            applicationObj.applicationDateObjs.add(applicationDateObj);
            labels.put(label, applicationObj);
            namesArray.list.add(label);
            return applicationDateObj.dateOfSaving;

        } else {
            ApplicationObj applicationObj = labels.get(label);
            byte[] representation = application.save();
            ApplicationDateObj applicationDateObj = new ApplicationDateObj(representation);
            applicationObj.applicationDateObjs.add(applicationDateObj);
            return applicationDateObj.dateOfSaving;
        }
    }
    private Reader reader = new Reader();
    private byte[] word = new byte[4];

    private int readInt(RandomAccessFile raf) throws IOException, EndOfBufferException {
        if (raf.read(word) == -1) {
            return -1;
        }
        return reader.setBytes(word).readInt();
    }

    private String readString(RandomAccessFile raf, int length) throws EndOfBufferException, IOException {
        byte[] buffer = new byte[length];
        raf.read(buffer);
        return reader.setBytes(buffer).readString(length);
    }

    private class Info {

        public Info(String name, String date, String symbolicName) {
            this.date = date;
            this.name = name;
            this.symbolicName = symbolicName;
        }
        public String date;
        public String name;
        public String symbolicName;
    }
    private List<Info> infosToRemove = new ArrayList<Info>();

    public void remove(String applicationName, String symbolicName) {
        ApplicationObj applicationObj = null;
        List<ApplicationDateObj> applicationDateObjs = new ArrayList<ApplicationDateObj>();
        List<String> dates = new ArrayList<String>();
        for (ApplicationObj fa : labels.values()) {
            if (applicationName.equals(fa.nameOfApplication)
                    && symbolicName.equals(fa.symbolicNameOfBundle)) {
                for (ApplicationDateObj fb : fa.applicationDateObjs) {
                    applicationObj = fa;
                    applicationDateObjs.add(fb);
                    dates.add(fb.dateOfSaving);
                }
            }
        }
        for (ApplicationDateObj applicationDateObj : applicationDateObjs) {
            if (applicationObj != null && applicationDateObj != null) {
                applicationObj.applicationDateObjs.remove(applicationDateObj);
                return;
            }
        }
        infosToRemove.add(new Info(applicationName, "ANY", symbolicName));
    }

    public void remove(String applicationName, String dateOfSaving, String symbolicName) {
        ApplicationObj applicationObj = null;
        ApplicationDateObj applicationDateObj = null;
        for (ApplicationObj fa : labels.values()) {
            if (applicationName.equals(fa.nameOfApplication)
                    && symbolicName.equals(fa.symbolicNameOfBundle)) {
                for (ApplicationDateObj fb : fa.applicationDateObjs) {
                    if (fb.dateOfSaving.equals(dateOfSaving)) {
                        applicationObj = fa;
                        applicationDateObj = fb;
                    }
                }
            }
        }
        if (applicationObj != null && applicationDateObj != null) {
            applicationObj.applicationDateObjs.remove(applicationDateObj);
            return;
        }
        infosToRemove.add(new Info(applicationName, dateOfSaving, symbolicName));
    }

    public byte[] load(String symbolicName, String applicationName, String dateOfSaving) {
        byte[] out = null;
        for (ApplicationObj applicationObj : labels.values()) {
            if (applicationName.equals(applicationObj.nameOfApplication)
                    && symbolicName.equals(applicationObj.symbolicNameOfBundle)) {
                for (ApplicationDateObj applicationDateObj : applicationObj.applicationDateObjs) {
                    if (applicationDateObj.dateOfSaving.equals(dateOfSaving)) {
                        return applicationDateObj.representation;
                    }
                }
            }
        }
        RandomAccessFile raf = null;
        File file = null;

        try {
            file = new File(path);
            if (file.exists()) {
                raf = new RandomAccessFile(file, "r");
                String comparedApplicationName = "";
                String comparedSymbolicName = "";
                String comparedDate = "";
                int sizeOfBlock = 0;
                boolean endOfFile = false;
                while ((!comparedApplicationName.equals(applicationName)
                        || !comparedDate.equals(dateOfSaving) || !comparedSymbolicName.equals(symbolicName))
                        && raf.getFilePointer() < raf.length()) {
                    raf.skipBytes(sizeOfBlock);
                    int length = readInt(raf);
                    if (length == -1) {
                        endOfFile = true;
                    } else {
                        comparedSymbolicName = readString(raf, length);
                        length = readInt(raf);
                        comparedApplicationName = readString(raf, length);
                        length = readInt(raf);
                        comparedDate = readString(raf, length);
                        length = readInt(raf);
                        sizeOfBlock = length;
                    }
                }
                out = new byte[sizeOfBlock];
                raf.read(out);
            }
        } catch (EndOfBufferException ex) {
            Logger.getLogger(ApplicationsIO.class.getName()).log(Level.SEVERE, null, ex);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(ApplicationsIO.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(ApplicationsIO.class.getName()).log(Level.SEVERE, null, ex);
        } finally {
            if (raf != null) {
                try {
                    raf.close();
                } catch (IOException ex) {
                    Logger.getLogger(ApplicationsIO.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        }
        return out;
    }

    private void saveApplicationObj(ApplicationObj applicationObj, ApplicationDateObj applicationDateObj, RandomAccessFile raf) throws IOException {
        Writer writer = new Writer();
        writer.write(applicationObj.symbolicNameOfBundle.length());
        writer.write(applicationObj.symbolicNameOfBundle);
        writer.write(applicationObj.nameOfApplication.length());
        writer.write(applicationObj.nameOfApplication);
        writer.write(applicationDateObj.dateOfSaving.length());
        writer.write(applicationDateObj.dateOfSaving);
        writer.write(applicationDateObj.representation.length);
        writer.write(applicationDateObj.representation);
        raf.write(writer.getBytes());
    }

    private boolean shouldBeRemoved(String name, String date, String symbolicName) {
        for (Info info : infosToRemove) {
            if ((info.date.equals(date) || info.date.equals("ANY")) && info.name.equals(name) && info.symbolicName.equals(symbolicName)) {
                return true;
            }
        }
        return false;
    }

    private void removeAllFromFile() {
        if (infosToRemove.size() == 0) {
            return;
        }
        RandomAccessFile raf = null;
        try {

            File file = null;
            Reader reader = new Reader();
            file = new File(path);
            raf = new RandomAccessFile(file, "r");
            byte[] all = new byte[(int) raf.length()];
            raf.read(all);
            reader.setBytes(all);
            raf.close();
            raf = null;
            Writer writer = new Writer();
            while (reader.getPosition() < reader.getSize()) {
                int length = reader.readInt();
                String symbolicName = reader.readString(length);
                length = reader.readInt();
                String name = reader.readString(length);
                length = reader.readInt();
                String date = reader.readString(length);
                length = reader.readInt();
                if (!shouldBeRemoved(name, date, symbolicName)) {

                    byte[] bytes = new byte[length];
                    reader.readBytes(bytes);
                    writer.write(symbolicName.length());
                    writer.write(symbolicName);
                    writer.write(name.length());
                    writer.write(name);
                    writer.write(date.length());
                    writer.write(date);
                    writer.write(bytes.length);
                    writer.write(bytes);
                } else {
                    reader.skip(length);
                }
            }
            file.delete();
            file.createNewFile();
            raf = new RandomAccessFile(file, "rw");
            raf.write(writer.getBytes());
            raf.close();
            raf = null;
        } catch (EndOfBufferException ex) {
            Logger.getLogger(ApplicationsIO.class.getName()).log(Level.SEVERE, null, ex);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(ApplicationsIO.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(ApplicationsIO.class.getName()).log(Level.SEVERE, null, ex);
        } finally {
            if (raf != null) {
                try {
                    raf.close();
                } catch (IOException ex) {
                    Logger.getLogger(ApplicationsIO.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        }
        infosToRemove.clear();
    }

    public void save() {
        RandomAccessFile raf = null;
        File file = null;
        try {
            file = new File(path);
            if (!file.exists()) {
                file.createNewFile();
            }
            raf = new RandomAccessFile(file, "rw");
            raf.seek(raf.length());
            for (ApplicationObj applicationObj : labels.values()) {
                for (ApplicationDateObj applicationDateObj : applicationObj.applicationDateObjs) {
                    saveApplicationObj(applicationObj, applicationDateObj, raf);
                }
            }
            labels.clear();
        } catch (FileNotFoundException ex) {
            Logger.getLogger(ApplicationsIO.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(ApplicationsIO.class.getName()).log(Level.SEVERE, null, ex);
        } finally {
            if (raf != null) {
                try {
                    raf.close();
                } catch (IOException ex) {
                    Logger.getLogger(ApplicationsIO.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        }
        removeAllFromFile();
    }
}
