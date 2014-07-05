package ogla.datafitting;


import ogla.datafitting.operations.ExpotentailFitting;
import ogla.datafitting.operations.GaussianFitting;
import ogla.datafitting.operations.LogFitting;
import ogla.datafitting.operations.PolynomialFitting;
import java.awt.event.ActionEvent;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JFrame;
import javax.swing.JSpinner;
import javax.swing.event.ListSelectionEvent;

public class DataFrame extends javax.swing.JFrame {

    private RenameFrame renameFrame = null;

    public DataFrame() {
        initComponents();
        jList1.setModel(listBundleData.model);
        jList2.setModel(listRepositoryData.model);
        jList3.setModel(listModification.model);
        DefaultComboBoxModel defaultComboBoxModel = new DefaultComboBoxModel();
        defaultComboBoxModel.addElement(new ExpotentailFitting(bundleData));
        defaultComboBoxModel.addElement(new LogFitting(bundleData));
        defaultComboBoxModel.addElement(new PolynomialFitting(bundleData));
        defaultComboBoxModel.addElement(new GaussianFitting(bundleData));
        jComboBox1.setModel(defaultComboBoxModel);
    }
    private ListBundleData listBundleData = new ListBundleData();
    private ListDataRepository listRepositoryData = new ListDataRepository();
    private ListDataRepository listModification = new ListDataRepository();

    public ListBundleData getListBundleData() {
        return listBundleData;
    }

    public ListDataRepository getListRepositoryData() {
        return listRepositoryData;
    }

    class DataBundleImpl implements DataBundle {

        public AnalysisEvent[] getEvents() {
            return null;
        }

        public int size() {
            return listModification.list.size();
        }

        public DataRepository get(int index) {
            return listModification.list.get(index);
        }

        public String getLabel() {
            return "Fitting Tools";
        }

        public String getDescription() {
            return "";
        }

        public JFrame getFrame() {
            return DataFrame.this;
        }

        public boolean canBeSaved() {
            return true;
        }

        public DataRepository load(byte[] bytes) {

            try {
                Operation.DataRepositoryImpl dataRepositoryImpl =
                        new Operation.DataRepositoryImpl(bundleData);
                Reader reader = new Reader(bytes);
                int version = reader.readInt();
                if (version == 0) {
                    int lengthOfLabel = reader.readInt();
                    String label = reader.readStr(lengthOfLabel);
                    int lengthOfDescription = reader.readInt();
                    String description = reader.readStr(lengthOfDescription);

                    int numberOfRepositories = reader.readInt();

                    for (int x = 0; x < numberOfRepositories; x++) {
                        Operation.DataBlockImpl dataBlockImpl = new Operation.DataBlockImpl(dataRepositoryImpl);
                        dataRepositoryImpl.dataBlocks.add(dataBlockImpl);
                        int numberOfRows = reader.readInt();
                        for (int fa = 0; fa < numberOfRows; fa++) {
                            int numberOfValues = reader.readInt();
                            Operation.DataRowImpl lineDataImpl = new Operation.DataRowImpl(numberOfValues);
                            for (int fb = 0; fb < numberOfValues; fb++) {
                                float v = reader.readFloat();
                                int length = reader.readInt();
                                String flabel = reader.readStr(length);
                                lineDataImpl.values[fb] = new Operation.DataRowImpl.ValueDataImpl(v, flabel);
                            }
                            dataBlockImpl.rows.add(lineDataImpl);
                        }
                        dataRepositoryImpl.setLabel(label);
                        dataRepositoryImpl.setDescription(description);
                    }
                }

                return dataRepositoryImpl;
            } catch (EndOfBufferException ex) {
                Logger.getLogger(DataFrame.class.getName()).log(Level.SEVERE, null, ex);
            } catch (IOException ex) {
                Logger.getLogger(DataFrame.class.getName()).log(Level.SEVERE, null, ex);
            }

            return null;

        }

        public void loadToContainer(byte[] bytes) {
            DataRepository repositoryData = load(bytes);
            listModification.add(repositoryData);
        }

        public void loadBundleData(byte[] bytes) {
        }

        public byte[] saveDataBundle() {
            return null;
        }

        public String getSymbolicName() {
            return "data_fitting_analysis_*(&%";
        }

        public DataBundleImpl() {
            root.documents.add(doc);
        }
        private Help.Document doc = new DefaultDocumentImpl("overview", "fitting_tools_help.html", DataBundleImpl.class);
        private DefaultChapterImpl root = new DefaultChapterImpl("Data: fitting tools");
        private Help help = new Help() {

            public Chapter getRootChapter() {
                return root;
            }
        };

        public Help getHelp() {
            return help;
        }
    }
    private DataBundleImpl bundleData = new DataBundleImpl();

    public DataBundle getBundleData() {
        return bundleData;
    }

    /** This method is called from within the constructor to
     * initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is
     * always regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jToolBar1 = new javax.swing.JToolBar();
        jComboBox1 = new javax.swing.JComboBox();
        jButton2 = new javax.swing.JButton();
        jSeparator1 = new javax.swing.JToolBar.Separator();
        jLabel2 = new javax.swing.JLabel();
        jSpinner1 = new javax.swing.JSpinner();
        jSeparator2 = new javax.swing.JToolBar.Separator();
        jLabel3 = new javax.swing.JLabel();
        jSpinner2 = new javax.swing.JSpinner();
        jScrollPane4 = new javax.swing.JScrollPane();
        jList3 = new javax.swing.JList();
        jButton4 = new javax.swing.JButton();
        jScrollPane1 = new javax.swing.JScrollPane();
        jList1 = new javax.swing.JList();
        jScrollPane2 = new javax.swing.JScrollPane();
        jList2 = new javax.swing.JList();
        jButton3 = new javax.swing.JButton();
        jButton1 = new javax.swing.JButton();
        jToolBar2 = new javax.swing.JToolBar();
        jLabel4 = new javax.swing.JLabel();
        jSpinner3 = new javax.swing.JSpinner();
        jSeparator5 = new javax.swing.JToolBar.Separator();
        jLabel5 = new javax.swing.JLabel();
        jSpinner4 = new javax.swing.JSpinner();
        jSeparator4 = new javax.swing.JToolBar.Separator();
        jLabel6 = new javax.swing.JLabel();
        jSpinner5 = new javax.swing.JSpinner();
        jSeparator3 = new javax.swing.JToolBar.Separator();
        jLabel7 = new javax.swing.JLabel();
        jSpinner6 = new javax.swing.JSpinner();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        jToolBar1.setFloatable(false);
        jToolBar1.setRollover(true);

        jComboBox1.addItemListener(new java.awt.event.ItemListener() {
            public void itemStateChanged(java.awt.event.ItemEvent evt) {
                jComboBox1ItemStateChanged(evt);
            }
        });
        jToolBar1.add(jComboBox1);

        jButton2.setText("Refresh");
        jButton2.setFocusable(false);
        jButton2.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jButton2.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        jToolBar1.add(jButton2);
        jToolBar1.add(jSeparator1);

        jLabel2.setText("X column");
        jToolBar1.add(jLabel2);

        jSpinner1.setModel(new javax.swing.SpinnerNumberModel(Integer.valueOf(0), Integer.valueOf(0), null, Integer.valueOf(1)));
        jToolBar1.add(jSpinner1);
        jToolBar1.add(jSeparator2);

        jLabel3.setText("Y column");
        jToolBar1.add(jLabel3);

        jSpinner2.setModel(new javax.swing.SpinnerNumberModel(Integer.valueOf(1), Integer.valueOf(0), null, Integer.valueOf(1)));
        jToolBar1.add(jSpinner2);

        jList3.setBorder(javax.swing.BorderFactory.createTitledBorder("Converted data"));
        jScrollPane4.setViewportView(jList3);

        jButton4.setText("Rename");
        jButton4.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton4ActionPerformed(evt);
            }
        });

        jList1.setBorder(javax.swing.BorderFactory.createTitledBorder("Bundles"));
        jList1.setSelectionMode(javax.swing.ListSelectionModel.SINGLE_INTERVAL_SELECTION);
        jList1.setLayoutOrientation(javax.swing.JList.VERTICAL_WRAP);
        jList1.addListSelectionListener(new javax.swing.event.ListSelectionListener() {
            public void valueChanged(javax.swing.event.ListSelectionEvent evt) {
                jList1ValueChanged(evt);
            }
        });
        jScrollPane1.setViewportView(jList1);

        jList2.setBorder(javax.swing.BorderFactory.createTitledBorder("Repositories"));
        jScrollPane2.setViewportView(jList2);

        jButton3.setText("Remove");
        jButton3.setFocusable(false);
        jButton3.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jButton3.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        jButton3.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton3ActionPerformed(evt);
            }
        });

        jButton1.setText("Convert");
        jButton1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton1ActionPerformed(evt);
            }
        });

        jToolBar2.setFloatable(false);
        jToolBar2.setRollover(true);

        jLabel4.setText("begin");
        jToolBar2.add(jLabel4);

        jSpinner3.setModel(new javax.swing.SpinnerNumberModel(Float.valueOf(0.0f), null, null, Float.valueOf(1.0f)));
        jToolBar2.add(jSpinner3);
        jToolBar2.add(jSeparator5);

        jLabel5.setText("end");
        jToolBar2.add(jLabel5);

        jSpinner4.setModel(new javax.swing.SpinnerNumberModel(Float.valueOf(10.0f), null, null, Float.valueOf(1.0f)));
        jToolBar2.add(jSpinner4);
        jToolBar2.add(jSeparator4);

        jLabel6.setText("step");
        jToolBar2.add(jLabel6);

        jSpinner5.setModel(new javax.swing.SpinnerNumberModel(Float.valueOf(0.1f), null, null, Float.valueOf(1.0f)));
        jToolBar2.add(jSpinner5);
        jToolBar2.add(jSeparator3);

        jLabel7.setText("degree");
        jToolBar2.add(jLabel7);

        jSpinner6.setModel(new javax.swing.SpinnerNumberModel(Float.valueOf(1.0f), Float.valueOf(0.0f), null, Float.valueOf(1.0f)));
        jToolBar2.add(jSpinner6);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(jScrollPane2, javax.swing.GroupLayout.DEFAULT_SIZE, 179, Short.MAX_VALUE)
                    .addComponent(jScrollPane1, javax.swing.GroupLayout.DEFAULT_SIZE, 179, Short.MAX_VALUE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(jScrollPane4, javax.swing.GroupLayout.DEFAULT_SIZE, 203, Short.MAX_VALUE)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(jButton3)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(jButton4)
                        .addGap(62, 62, 62))
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(jButton1)
                        .addContainerGap())))
            .addComponent(jToolBar1, javax.swing.GroupLayout.DEFAULT_SIZE, 400, Short.MAX_VALUE)
            .addComponent(jToolBar2, javax.swing.GroupLayout.DEFAULT_SIZE, 400, Short.MAX_VALUE)
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(jToolBar1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jToolBar2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(17, 17, 17)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(jScrollPane1, javax.swing.GroupLayout.PREFERRED_SIZE, 97, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(jScrollPane2, javax.swing.GroupLayout.DEFAULT_SIZE, 112, Short.MAX_VALUE))
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                            .addComponent(jButton4)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(jButton1)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(jButton3)))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(jScrollPane4, javax.swing.GroupLayout.DEFAULT_SIZE, 143, Short.MAX_VALUE))))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void jButton4ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton4ActionPerformed
        int index = jList3.getSelectedIndex();
        if (index == -1) {
            return;
        }
        if (renameFrame == null) {
            renameFrame = new RenameFrame();
        }
        renameFrame.setVisible(true);
        Operation.DataRepositoryImpl dataRepositoriesGroupImpl = (Operation.DataRepositoryImpl) listModification.list.get(index);
        renameFrame.setDataRepositoriesGroupImpl(dataRepositoriesGroupImpl);
}//GEN-LAST:event_jButton4ActionPerformed

    private void jList1ValueChanged(javax.swing.event.ListSelectionEvent evt) {//GEN-FIRST:event_jList1ValueChanged
        int index = jList1.getSelectedIndex();
        if (index == -1) {
            return;
        }
        DataBundle bundleData = listBundleData.list.get(index);
        DataRepository[] repositories = new DataRepository[bundleData.size()];
        for (int fa = 0; fa < bundleData.size(); fa++) {
            repositories[fa] = bundleData.get(fa);
        }
        getListRepositoryData().setArray(repositories);
}//GEN-LAST:event_jList1ValueChanged

    private void jButton3ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton3ActionPerformed
        int index = jList3.getSelectedIndex();
        if (index == -1) {
            return;
        }
        listModification.remove(index);
}//GEN-LAST:event_jButton3ActionPerformed

    private float getValue(JSpinner spinner) {
        Number n = (Number) spinner.getValue();
        return n.floatValue();
    }

    private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton1ActionPerformed
        int index = jList2.getSelectedIndex();
        if (index == -1) {
            return;
        }
        DataRepository dataRepository = listRepositoryData.list.get(index);
        Operation operation = (Operation) jComboBox1.getSelectedItem();
        Operation.DataRepositoryImpl dataRepositoryImpl = new Operation.DataRepositoryImpl(bundleData);
        dataRepositoryImpl.setLabel(dataRepository.getLabel() + "_" + operation.getSuffix());
        float begin = getValue(jSpinner3);
        float end = getValue(jSpinner4);
        float step = getValue(jSpinner5);
        float amplodegree = getValue(jSpinner6);
        for (int fa = 0; fa < dataRepository.size(); fa++) {
            DataBlock outcomes = operation.modify(dataRepository.get(fa), dataRepositoryImpl, begin, end, step, amplodegree,
                    ((Number) jSpinner1.getValue()).intValue(), ((Number) jSpinner2.getValue()).intValue());
            dataRepositoryImpl.dataBlocks.add(outcomes);
            this.listModification.add(dataRepositoryImpl);
        }
}//GEN-LAST:event_jButton1ActionPerformed

    private void jComboBox1ItemStateChanged(java.awt.event.ItemEvent evt) {//GEN-FIRST:event_jComboBox1ItemStateChanged
        Operation operation = (Operation) jComboBox1.getSelectedItem();
        if (operation instanceof PolynomialFitting || operation instanceof GaussianFitting) {
            jSpinner6.setEnabled(true);
        } else {
            jSpinner6.setEnabled(false);
        }
    }//GEN-LAST:event_jComboBox1ItemStateChanged

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        java.awt.EventQueue.invokeLater(new Runnable() {

            public void run() {
                new DataFrame().setVisible(true);
            }
        });
    }
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton jButton1;
    private javax.swing.JButton jButton2;
    private javax.swing.JButton jButton3;
    private javax.swing.JButton jButton4;
    private javax.swing.JComboBox jComboBox1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JLabel jLabel3;
    private javax.swing.JLabel jLabel4;
    private javax.swing.JLabel jLabel5;
    private javax.swing.JLabel jLabel6;
    private javax.swing.JLabel jLabel7;
    private javax.swing.JList jList1;
    private javax.swing.JList jList2;
    private javax.swing.JList jList3;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JScrollPane jScrollPane2;
    private javax.swing.JScrollPane jScrollPane4;
    private javax.swing.JToolBar.Separator jSeparator1;
    private javax.swing.JToolBar.Separator jSeparator2;
    private javax.swing.JToolBar.Separator jSeparator3;
    private javax.swing.JToolBar.Separator jSeparator4;
    private javax.swing.JToolBar.Separator jSeparator5;
    private javax.swing.JSpinner jSpinner1;
    private javax.swing.JSpinner jSpinner2;
    private javax.swing.JSpinner jSpinner3;
    private javax.swing.JSpinner jSpinner4;
    private javax.swing.JSpinner jSpinner5;
    private javax.swing.JSpinner jSpinner6;
    private javax.swing.JToolBar jToolBar1;
    private javax.swing.JToolBar jToolBar2;
    // End of variables declaration//GEN-END:variables
}
