package ogla.datacsv;

import ogla.core.Help;
import ogla.core.event.AnalysisEvent;
import ogla.core.util.DefaultChapterImpl;
import ogla.core.util.DefaultDocumentImpl;
import ogla.core.util.Reader;
import ogla.core.util.Reader.EndOfBufferException;
import java.io.FileNotFoundException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.DefaultListModel;
import java.awt.event.MouseEvent;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.filechooser.FileNameExtensionFilter;
import ogla.core.data.DataBundle;
import ogla.core.data.DataRepository;
import ogla.core.data.listeners.RepositoryListener;
import ogla.core.data.utils.DataTools;

/**
 *
 * @author Marcin
 */
public class DataFrame extends javax.swing.JFrame {

    private DefaultListModel listModel = new DefaultListModel();
    private JFileChooser fileChooser = new JFileChooser();
    private Parser parser = new Parser();
    private DataBundle dataBundle = new DataBundleImpl();

    public DataBundle getDataBundle() {
        return dataBundle;
    }

    private class DataBundleImpl implements DataBundle {

        private DefaultChapterImpl root = new DefaultChapterImpl("Data CSV");

        public DataBundleImpl() {
            root.documents.add(new DefaultDocumentImpl("Overview", "data_csv_help.html", DataFrame.class));
        }

        public DataRepository get(int index) {
            return listDataRepository.get(index);
        }

        public int size() {
            return listDataRepository.size();
        }

        public boolean canBeSaved() {
            return true;
        }

        public void loadBundleData(byte[] bytes) {
        }

        public byte[] saveDataBundle() {
            return null;
        }

        public void loadToContainer(byte[] bytes) {
            listDataRepository.add(this.load(bytes));
        }

        public DataRepository load(byte[] bytes) {
            try {
                Reader reader = new Reader(bytes);
                int version = reader.readInt();
                if (version == 0) {
                    DataRepositoryImpl dataRepositoryImpl = new DataRepositoryImpl(DataBundleImpl.this);
                    String label = reader.readString(reader.readInt());
                    String path = reader.readString(reader.readInt());
                    dataRepositoryImpl.set(label, path);
                    int size = reader.readInt();
                    for (int fa = 0; fa < size; fa++) {
                        DataTools.DataRowImpl dataRow = new DataTools.DataRowImpl();
                        dataRepositoryImpl.dataBlockImpl.rows.add(dataRow);
                        int size1 = reader.readInt();

                        for (int fc = 0; fc < size1; fc++) {
                            label = reader.readString(reader.readInt());
                            double number = reader.readDouble();
                            DataValueImpl dataValueImpl = new DataValueImpl(number, label);
                            dataRow.add(dataValueImpl);

                        }
                    }
                    return dataRepositoryImpl;
                }
            } catch (EndOfBufferException ex) {
                Logger.getLogger(DataFrame.class.getName()).log(Level.SEVERE, null, ex);
            } catch (IOException ex) {
                Logger.getLogger(DataFrame.class.getName()).log(Level.SEVERE, null, ex);
            }
            return null;
        }

        public String getSymbolicName() {
            return "analysis_data_csv_(*&^^&";
        }

        public String getLabel() {
            return "Data CSV";
        }

        public String getDescription() {
            return "";
        }

        public JFrame getFrame() {
            return DataFrame.this;
        }

        public AnalysisEvent[] getEvents() {
            return null;
        }

        public Help getHelp() {
            return new Help() {

                public Chapter getRootChapter() {
                    return root;
                }
            };
        }
    }

    private class ListDataRepositoriesGroup extends ArrayList<DataRepository> {

        private int isSameName(String label) {
            for (int fa = 0; fa < this.size(); fa++) {
                if (label.equals(this.get(fa).getLabel())) {
                    return fa;
                }
            }
            return -1;
        }

        private void checkName(DataRepository dataRepository) {
            DataRepositoryImpl dataRepositoryImpl = (DataRepositoryImpl) dataRepository;
            int d = 1;
            int index = 0;
            StringBuilder nlabel = new StringBuilder(dataRepositoryImpl.getLabel());
            if ((index = isSameName(nlabel.toString())) != -1) {
                nlabel.append("_1");
                while ((index = isSameName(nlabel.toString())) != -1) {
                    d++;
                    nlabel = nlabel.replace(nlabel.length() - 1, nlabel.length(), String.valueOf(d));
                }
            }
            dataRepositoryImpl.setLabel(nlabel.toString());
        }

        @Override
        public boolean add(DataRepository dataRepositoriesGroup) {
            checkName(dataRepositoriesGroup);
            boolean b = super.add(dataRepositoriesGroup);
            listModel.addElement(dataRepositoriesGroup);
            return b;
        }

        @Override
        public boolean remove(Object object) {
            boolean b = false;
            if (object instanceof DataRepository) {
                DataRepository sdi = (DataRepository) object;
                b = super.remove(sdi);
                listModel.removeElement(sdi);
            }
            return b;
        }
    };
    private ListDataRepositoriesGroup listDataRepository = new ListDataRepositoriesGroup();

    private class Parser {

        DataRepositoryImpl dataRepositoryImpl = null;
        File file;

        private Parser() {
        }

        private final String removeWhiteSpace(String text) {
            Pattern pattern = Pattern.compile("\\s+");
            Matcher matcher = pattern.matcher(text);
            text = matcher.replaceAll("");
            return text;
        }

        private final void parseLine(String line, List<DataTools.DataRowImpl> row) throws NumberFormatException {

            DataTools.DataRowImpl dataRow = new DataTools.DataRowImpl();
            final String[] values = line.split(",");
            final Number[] numbers = new Number[values.length];
            for (int fa = 0; fa < numbers.length; fa++) {
                try {
                    dataRow.add(new DataValueImpl(new Float(values[fa]), values[fa]));
                } catch (NumberFormatException ex) {
                    throw ex;
                }
            }
            row.add(dataRow);
        }
        private RandomAccessFile randomAccessFile = null;

        private final void parse(File file) {
            this.file = file;
            dataRepositoryImpl = new DataRepositoryImpl(file, dataBundle);
            try {
                randomAccessFile = new RandomAccessFile(file, "rw");
                byte[] content = new byte[(int)randomAccessFile.length()];
                randomAccessFile.read(content);
                String context = new String(content);
                String[] lines = context.split("\n");
                for (String line : lines) {
                    parseLine(line, dataRepositoryImpl.dataBlockImpl.rows);
                }
                DataTools.sortAsc(dataRepositoryImpl.dataBlockImpl.rows);
                listDataRepository.add(dataRepositoryImpl);
            } catch (FileNotFoundException ex) {
                Logger.getLogger(DataFrame.class.getName()).log(Level.SEVERE, null, ex);
            } catch (IOException ex) {
                Logger.getLogger(DataFrame.class.getName()).log(Level.SEVERE, null, ex);
            } catch (NumberFormatException ex) {
                JOptionPane.showMessageDialog(DataFrame.this,
                        "In file are stored values which are not numbers.",
                        "Error in file's content.", JOptionPane.ERROR_MESSAGE);
                Logger.getLogger(DataFrame.class.getName()).log(Level.SEVERE, null, ex);
            } finally {
                try {
                    randomAccessFile.close();
                } catch (IOException ex) {
                    Logger.getLogger(DataFrame.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        }
    }

    public DataFrame() {
        initComponents();
        jList1.setModel(listModel);
        fileChooser.setMultiSelectionEnabled(true);
        fileChooser.setFileFilter(new FileNameExtensionFilter(
                "CSV File", "csv"));
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jSplitPane1 = new javax.swing.JSplitPane();
        jPanel1 = new javax.swing.JPanel();
        jScrollPane3 = new javax.swing.JScrollPane();
        jToolBar1 = new javax.swing.JToolBar();
        jButton1 = new javax.swing.JButton();
        jButton3 = new javax.swing.JButton();
        jButton2 = new javax.swing.JButton();
        jButton4 = new javax.swing.JButton();
        jPanel2 = new javax.swing.JPanel();
        jScrollPane2 = new javax.swing.JScrollPane();
        jList1 = new javax.swing.JList(){
            @Override
            public String getToolTipText(MouseEvent evt) {
                int index = locationToIndex(evt.getPoint());
                String tip="";
                if(index!=-1){
                    Object item = getModel().getElementAt(index);
                    DataRepositoryImpl rdi=(DataRepositoryImpl) item;
                    tip=rdi.absolutePath;
                }
                return tip;
            }
        };

        javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 360, Short.MAX_VALUE)
        );
        jPanel1Layout.setVerticalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 253, Short.MAX_VALUE)
        );

        jSplitPane1.setLeftComponent(jPanel1);

        jScrollPane3.setMinimumSize(new java.awt.Dimension(130, 23));

        setDefaultCloseOperation(javax.swing.WindowConstants.DISPOSE_ON_CLOSE);
        setTitle("Data CSV");

        jToolBar1.setFloatable(false);
        jToolBar1.setRollover(true);

        jButton1.setText("Open File");
        jButton1.setFocusable(false);
        jButton1.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jButton1.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        jButton1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton1ActionPerformed(evt);
            }
        });
        jToolBar1.add(jButton1);

        jButton3.setText("Refresh");
        jButton3.setFocusable(false);
        jButton3.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jButton3.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        jButton3.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton3ActionPerformed(evt);
            }
        });
        jToolBar1.add(jButton3);

        jButton2.setText("Remove");
        jButton2.setFocusable(false);
        jButton2.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jButton2.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        jButton2.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton2ActionPerformed(evt);
            }
        });
        jToolBar1.add(jButton2);

        jButton4.setText("Rename");
        jButton4.setFocusable(false);
        jButton4.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jButton4.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        jButton4.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton4ActionPerformed(evt);
            }
        });
        jToolBar1.add(jButton4);

        jList1.setModel(listModel);
        jList1.setSelectionMode(javax.swing.ListSelectionModel.SINGLE_INTERVAL_SELECTION);
        jList1.setLayoutOrientation(javax.swing.JList.VERTICAL_WRAP);
        jScrollPane2.setViewportView(jList1);

        javax.swing.GroupLayout jPanel2Layout = new javax.swing.GroupLayout(jPanel2);
        jPanel2.setLayout(jPanel2Layout);
        jPanel2Layout.setHorizontalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 405, Short.MAX_VALUE)
            .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                .addComponent(jScrollPane2, javax.swing.GroupLayout.DEFAULT_SIZE, 405, Short.MAX_VALUE))
        );
        jPanel2Layout.setVerticalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 309, Short.MAX_VALUE)
            .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                .addComponent(jScrollPane2, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, 309, Short.MAX_VALUE))
        );

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jToolBar1, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, 405, Short.MAX_VALUE)
            .addComponent(jPanel2, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(jToolBar1, javax.swing.GroupLayout.PREFERRED_SIZE, 25, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jPanel2, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
    }// </editor-fold>//GEN-END:initComponents

    private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton1ActionPerformed

        int returnVal = fileChooser.showOpenDialog(this);
        if (returnVal == JFileChooser.APPROVE_OPTION) {
            File[] files = fileChooser.getSelectedFiles();
            if (files.length != 0) {
                for (int fa = 0; fa< files.length; fa++) {
                    parser.parse(files[fa]);
                }
            } else {
                parser.parse(fileChooser.getSelectedFile());
            }
        }
    }//GEN-LAST:event_jButton1ActionPerformed

    private void jButton3ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton3ActionPerformed
        if (jList1.getSelectedIndex() == -1) {
            return;
        }
        DataRepositoryImpl sdi = (DataRepositoryImpl) jList1.getSelectedValue();
        parser.parse(new File(sdi.absolutePath));
        DataRepository tsdi = parser.dataRepositoryImpl;
        listDataRepository.set(listDataRepository.indexOf(sdi), tsdi);
        for (int fa = 0; fa < tsdi.size(); fa++) {
            DataBlockImpl dataBlockImpl = (DataBlockImpl) tsdi.get(fa);
            for (RepositoryListener rl : dataBlockImpl.listeners) {
                rl.isChangedData(tsdi.get(0));
            }
        }

    }//GEN-LAST:event_jButton3ActionPerformed

    private void jButton2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton2ActionPerformed
        if (jList1.getSelectedIndex() == -1) {
            return;
        }
        DataRepository dataRepositoryImpl = (DataRepository) jList1.getSelectedValue();
        listDataRepository.remove(dataRepositoryImpl);
    }//GEN-LAST:event_jButton2ActionPerformed
    private RenameFrame renameFrame = null;
    private void jButton4ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton4ActionPerformed
        if (renameFrame == null) {
            renameFrame = new RenameFrame();
        }
        if (jList1.getSelectedIndex() == -1) {
            return;
        }
        DataRepositoryImpl dataRepositoryImpl = (DataRepositoryImpl) jList1.getSelectedValue();
        renameFrame.setDataRepositoriesGroup(dataRepositoryImpl);
    }//GEN-LAST:event_jButton4ActionPerformed
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton jButton1;
    private javax.swing.JButton jButton2;
    private javax.swing.JButton jButton3;
    private javax.swing.JButton jButton4;
    private javax.swing.JList jList1;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JPanel jPanel2;
    private javax.swing.JScrollPane jScrollPane2;
    private javax.swing.JScrollPane jScrollPane3;
    private javax.swing.JSplitPane jSplitPane1;
    private javax.swing.JToolBar jToolBar1;
    // End of variables declaration//GEN-END:variables
}
