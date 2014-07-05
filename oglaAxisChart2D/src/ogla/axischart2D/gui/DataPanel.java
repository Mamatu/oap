package ogla.axischart2D.gui;

import ogla.axischart.Displayer;
import ogla.axischart.gui.TableFrame;
import ogla.axischart.lists.ListDataBundle;
import ogla.axischart.lists.ListDataRepository;
import ogla.axischart.lists.ListDisplayer;
import ogla.axischart.lists.ListListener;
import ogla.axischart.lists.ListsContainer;
import ogla.axischart2D.AxisChart2D;
import ogla.axischart2D.DisplayerImpl;
import ogla.axischart2D.plugin.plotmanager.AxisChart2DPlotManager;
import ogla.axischart2D.plugin.plotmanager.AxisChart2DPlotManagerBundle;
import ogla.axischart2D.util.DisplayerTools;
import ogla.axischart2D.lists.PlotManagerBundleList;
import ogla.axischart2D.util.DisplayerExtendedImpl;
import ogla.axischart2D.util.AxisIndices;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.DefaultComboBoxModel;
import javax.swing.DefaultListModel;
import javax.swing.JFrame;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPopupMenu;
import ogla.core.util.ListElementManager;
import ogla.core.data.DataBlock;
import ogla.core.data.DataBundle;
import ogla.core.data.DataRepository;

public final class DataPanel extends javax.swing.JPanel {

    protected AxisChart2D axisChart;
    protected ListsContainer listsContainer;
    protected DefaultComboBoxModel defaultComboBoxModel = new DefaultComboBoxModel();
    protected DefaultListModel defaultListModel3 = new DefaultListModel();
    protected DefaultListModel defaultListModel1 = new DefaultListModel();
    protected DefaultListModel defaultListModel2 = new DefaultListModel();
    private ListDisplayer listDisplayer = null;
    private ListDataRepository listDataRepository = null;
    private ListDataBundle listDataBundle = null;
    private PlotManagerBundleList listPlotManagerBundle = null;

    /** Creates new form DataPanel */
    public DataPanel(AxisChart2D axisChart, ListsContainer listsContainer) {
        initComponents();
        jLabel2.setVisible(false);
        this.axisChart = axisChart;
        this.listsContainer = listsContainer;
        jList1.setModel(defaultListModel1);
        jList2.setModel(defaultListModel2);
        jList3.setModel(defaultListModel3);
        jComboBox1.setModel(defaultComboBoxModel);
        this.listPlotManagerBundle = listsContainer.get(PlotManagerBundleList.class);
        this.listPlotManagerBundle.addListListener(new ListListener<AxisChart2DPlotManagerBundle>() {

            @Override
            public void isAdded(AxisChart2DPlotManagerBundle t) {
                defaultComboBoxModel.addElement(ListElementManager.register(t));
            }

            @Override
            public void isRemoved(AxisChart2DPlotManagerBundle t) {
                defaultComboBoxModel.removeElementAt(ListElementManager.getIndex(t, defaultComboBoxModel));
            }
        });


        this.listDataBundle = listsContainer.get(ListDataBundle.class);
        this.listDataBundle.addListListener(
                new ListListener<DataBundle>() {

                    @Override
                    public void isAdded(DataBundle t) {
                        defaultListModel1.addElement(ListElementManager.register(t));
                    }

                    @Override
                    public void isRemoved(DataBundle t) {
                        defaultListModel1.removeElementAt(ListElementManager.getIndex(t, defaultListModel1));
                    }
                });

        this.listDataRepository = listsContainer.get(ListDataRepository.class);
        this.listDataRepository.addListListener(new ListListener<DataRepository>() {

            @Override
            public void isAdded(DataRepository t) {
                defaultListModel2.addElement(ListElementManager.register(t));
            }

            @Override
            public void isRemoved(DataRepository t) {
                defaultListModel2.removeElementAt(ListElementManager.getIndex(t, defaultListModel2));
            }
        });

        this.listDisplayer = listsContainer.get(ListDisplayer.class);
        this.listDisplayer.addListListener(new ListListener<Displayer>() {

            @Override
            public void isAdded(Displayer t) {
                defaultListModel3.addElement(ListElementManager.register(t));
            }

            @Override
            public void isRemoved(Displayer t) {
                defaultListModel3.removeElementAt(ListElementManager.getIndex(t, defaultListModel3));
            }
        });
    }
    private Map<DataBundle, List<DataRepository>> bundles = new HashMap<DataBundle, List<DataRepository>>();

    public void fillList(DataBundle dataBundle) {
        if (bundles.containsKey(dataBundle)) {
            List<DataRepository> dataRepositorys = bundles.get(dataBundle);
            for (int fa = 0; fa < dataRepositorys.size(); fa++) {
                DataRepository dataRepository = dataRepositorys.get(fa);
                this.listDataRepository.remove(dataRepository);
            }
            dataRepositorys.clear();

            for (int fa = 0; fa < dataBundle.size(); fa++) {
                DataRepository dataRepository = dataBundle.get(fa);
                this.listDataRepository.add(dataRepository);
                dataRepositorys.add(dataRepository);
            }

        } else {
            List<DataRepository> dataRepositorys = new ArrayList<DataRepository>();
            for (int fa = 0; fa < dataBundle.size(); fa++) {
                DataRepository dataRepository = dataBundle.get(fa);
                this.listDataRepository.add(dataRepository);
                dataRepositorys.add(dataRepository);
            }
            bundles.put(dataBundle, dataRepositorys);
        }
    }

    /** This method is called from within the constructor to
     * initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is
     * always regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jPanel2 = new javax.swing.JPanel();
        jSplitPane1 = new javax.swing.JSplitPane();
        jPanel1 = new javax.swing.JPanel();
        jToolBar3 = new javax.swing.JToolBar();
        jButton1 = new javax.swing.JButton();
        jScrollPane1 = new javax.swing.JScrollPane();
        jList1 = new javax.swing.JList();
        jPanel4 = new javax.swing.JPanel();
        jScrollPane2 = new javax.swing.JScrollPane();
        jList2 = new javax.swing.JList();
        jToolBar1 = new javax.swing.JToolBar();
        jButton2 = new javax.swing.JButton();
        jCheckBoxAsGroup = new javax.swing.JCheckBox();
        jLabel1 = new javax.swing.JLabel();
        jPanel3 = new javax.swing.JPanel();
        jScrollPane3 = new javax.swing.JScrollPane();
        jList3 = new javax.swing.JList();
        jToolBar2 = new javax.swing.JToolBar();
        jButton5 = new javax.swing.JButton();
        jComboBox1 = new javax.swing.JComboBox();
        jButton3 = new javax.swing.JButton();
        jButton4 = new javax.swing.JButton();
        jLabel2 = new javax.swing.JLabel();
        jToolBar4 = new javax.swing.JToolBar();
        jButton6 = new javax.swing.JButton();
        jButton8 = new javax.swing.JButton();
        jButton7 = new javax.swing.JButton();
        jSeparator1 = new javax.swing.JToolBar.Separator();
        jButton9 = new javax.swing.JButton();
        jButton10 = new javax.swing.JButton();
        jSeparator2 = new javax.swing.JToolBar.Separator();
        jSpinner1 = new javax.swing.JSpinner();
        jSeparator3 = new javax.swing.JToolBar.Separator();
        jCheckBox1 = new javax.swing.JCheckBox();
        jToolBar5 = new javax.swing.JToolBar();
        jLabel3 = new javax.swing.JLabel();
        jSpinner2 = new javax.swing.JSpinner();
        jComboBox2 = new javax.swing.JComboBox();
        jSeparator4 = new javax.swing.JToolBar.Separator();
        jLabel4 = new javax.swing.JLabel();
        jSpinner3 = new javax.swing.JSpinner();
        jComboBox3 = new javax.swing.JComboBox();

        setPreferredSize(new java.awt.Dimension(670, 340));

        jPanel2.setBorder(javax.swing.BorderFactory.createTitledBorder("Bundles of repositories"));

        jPanel1.setPreferredSize(new java.awt.Dimension(192, 394));

        jToolBar3.setFloatable(false);
        jToolBar3.setRollover(true);

        jButton1.setText("Execute");
        jButton1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton1ActionPerformed(evt);
            }
        });
        jToolBar3.add(jButton1);

        jList1.setBorder(new javax.swing.border.SoftBevelBorder(javax.swing.border.BevelBorder.RAISED));
        jScrollPane1.setViewportView(jList1);

        javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jToolBar3, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addComponent(jScrollPane1, javax.swing.GroupLayout.DEFAULT_SIZE, 72, Short.MAX_VALUE)
        );
        jPanel1Layout.setVerticalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addComponent(jToolBar3, javax.swing.GroupLayout.PREFERRED_SIZE, 25, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jScrollPane1, javax.swing.GroupLayout.DEFAULT_SIZE, 282, Short.MAX_VALUE))
        );

        jSplitPane1.setLeftComponent(jPanel1);

        jList2.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseReleased(java.awt.event.MouseEvent evt) {
                jList2MouseReleased(evt);
            }
        });
        jScrollPane2.setViewportView(jList2);

        jToolBar1.setFloatable(false);
        jToolBar1.setRollover(true);

        jButton2.setText("Display");
        jButton2.setFocusable(false);
        jButton2.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jButton2.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        jButton2.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton2ActionPerformed(evt);
            }
        });
        jToolBar1.add(jButton2);

        jCheckBoxAsGroup.setFocusable(false);
        jCheckBoxAsGroup.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jToolBar1.add(jCheckBoxAsGroup);

        jLabel1.setText("as group");
        jToolBar1.add(jLabel1);

        javax.swing.GroupLayout jPanel4Layout = new javax.swing.GroupLayout(jPanel4);
        jPanel4.setLayout(jPanel4Layout);
        jPanel4Layout.setHorizontalGroup(
            jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jToolBar1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addComponent(jScrollPane2, javax.swing.GroupLayout.DEFAULT_SIZE, 143, Short.MAX_VALUE)
        );
        jPanel4Layout.setVerticalGroup(
            jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel4Layout.createSequentialGroup()
                .addComponent(jToolBar1, javax.swing.GroupLayout.PREFERRED_SIZE, 25, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jScrollPane2, javax.swing.GroupLayout.DEFAULT_SIZE, 282, Short.MAX_VALUE))
        );

        jSplitPane1.setRightComponent(jPanel4);

        javax.swing.GroupLayout jPanel2Layout = new javax.swing.GroupLayout(jPanel2);
        jPanel2.setLayout(jPanel2Layout);
        jPanel2Layout.setHorizontalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jSplitPane1, javax.swing.GroupLayout.DEFAULT_SIZE, 221, Short.MAX_VALUE)
        );
        jPanel2Layout.setVerticalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jSplitPane1, javax.swing.GroupLayout.DEFAULT_SIZE, 313, Short.MAX_VALUE)
        );

        jPanel3.setBorder(javax.swing.BorderFactory.createTitledBorder("Displayed data"));

        jList3.setBorder(javax.swing.BorderFactory.createTitledBorder("Displayed data"));
        jList3.addListSelectionListener(new javax.swing.event.ListSelectionListener() {
            public void valueChanged(javax.swing.event.ListSelectionEvent evt) {
                jList3ValueChanged(evt);
            }
        });
        jScrollPane3.setViewportView(jList3);

        jToolBar2.setFloatable(false);
        jToolBar2.setForeground(new java.awt.Color(1, 1, 1));
        jToolBar2.setRollover(true);

        jButton5.setText("Change");
        jButton5.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton5ActionPerformed(evt);
            }
        });
        jToolBar2.add(jButton5);

        jComboBox1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jComboBox1ActionPerformed(evt);
            }
        });
        jToolBar2.add(jComboBox1);

        jButton3.setText("Hide");
        jButton3.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton3ActionPerformed(evt);
            }
        });
        jToolBar2.add(jButton3);

        jButton4.setText("Configure");
        jButton4.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton4ActionPerformed(evt);
            }
        });
        jToolBar2.add(jButton4);

        jLabel2.setForeground(new java.awt.Color(231, 28, 28));
        jLabel2.setText("Another frame is activated.");
        jToolBar2.add(jLabel2);

        jToolBar4.setFloatable(false);
        jToolBar4.setRollover(true);

        jButton6.setText("<");
        jButton6.setFocusable(false);
        jButton6.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jButton6.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        jButton6.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton6ActionPerformed(evt);
            }
        });
        jToolBar4.add(jButton6);

        jButton8.setText("Return");
        jButton8.setToolTipText("Return to the most current repository");
        jButton8.setFocusable(false);
        jButton8.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jButton8.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        jButton8.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton8ActionPerformed(evt);
            }
        });
        jToolBar4.add(jButton8);

        jButton7.setText(">");
        jButton7.setFocusable(false);
        jButton7.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jButton7.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        jButton7.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton7ActionPerformed(evt);
            }
        });
        jToolBar4.add(jButton7);
        jToolBar4.add(jSeparator1);

        jButton9.setText(">>");
        jButton9.setToolTipText("Play");
        jButton9.setFocusable(false);
        jButton9.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jButton9.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        jButton9.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton9ActionPerformed(evt);
            }
        });
        jToolBar4.add(jButton9);

        jButton10.setText("||");
        jButton10.setFocusable(false);
        jButton10.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jButton10.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        jButton10.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton10ActionPerformed(evt);
            }
        });
        jToolBar4.add(jButton10);
        jToolBar4.add(jSeparator2);

        jSpinner1.setModel(new javax.swing.SpinnerNumberModel(Float.valueOf(1.0f), Float.valueOf(1.0f), Float.valueOf(20.0f), Float.valueOf(1.0f)));
        jSpinner1.setToolTipText("Delay (in second) between next played history element. ");
        jToolBar4.add(jSpinner1);
        jToolBar4.add(jSeparator3);

        jCheckBox1.setText("use for all repositories");
        jCheckBox1.setFocusable(false);
        jCheckBox1.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jCheckBox1.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        jToolBar4.add(jCheckBox1);

        jToolBar5.setFloatable(false);
        jToolBar5.setRollover(true);

        jLabel3.setText("X column");
        jToolBar5.add(jLabel3);

        jSpinner2.setModel(new javax.swing.SpinnerNumberModel(Integer.valueOf(1), Integer.valueOf(1), null, Integer.valueOf(1)));
        jSpinner2.addChangeListener(new javax.swing.event.ChangeListener() {
            public void stateChanged(javax.swing.event.ChangeEvent evt) {
                jSpinner2StateChanged(evt);
            }
        });
        jToolBar5.add(jSpinner2);

        jComboBox2.setModel(new javax.swing.DefaultComboBoxModel(new String[] { "From begin", "From end" }));
        jComboBox2.addItemListener(new java.awt.event.ItemListener() {
            public void itemStateChanged(java.awt.event.ItemEvent evt) {
                jComboBox2ItemStateChanged(evt);
            }
        });
        jComboBox2.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jComboBox2ActionPerformed(evt);
            }
        });
        jToolBar5.add(jComboBox2);
        jToolBar5.add(jSeparator4);

        jLabel4.setText("Y column");
        jToolBar5.add(jLabel4);

        jSpinner3.setModel(new javax.swing.SpinnerNumberModel(Integer.valueOf(1), Integer.valueOf(1), null, Integer.valueOf(1)));
        jSpinner3.addChangeListener(new javax.swing.event.ChangeListener() {
            public void stateChanged(javax.swing.event.ChangeEvent evt) {
                jSpinner3StateChanged(evt);
            }
        });
        jToolBar5.add(jSpinner3);

        jComboBox3.setModel(new javax.swing.DefaultComboBoxModel(new String[] { "From begin", "From end" }));
        jComboBox3.setSelectedIndex(1);
        jComboBox3.addItemListener(new java.awt.event.ItemListener() {
            public void itemStateChanged(java.awt.event.ItemEvent evt) {
                jComboBox3ItemStateChanged(evt);
            }
        });
        jToolBar5.add(jComboBox3);

        javax.swing.GroupLayout jPanel3Layout = new javax.swing.GroupLayout(jPanel3);
        jPanel3.setLayout(jPanel3Layout);
        jPanel3Layout.setHorizontalGroup(
            jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jToolBar2, javax.swing.GroupLayout.DEFAULT_SIZE, 501, Short.MAX_VALUE)
            .addComponent(jToolBar4, javax.swing.GroupLayout.DEFAULT_SIZE, 501, Short.MAX_VALUE)
            .addComponent(jToolBar5, javax.swing.GroupLayout.DEFAULT_SIZE, 501, Short.MAX_VALUE)
            .addComponent(jScrollPane3, javax.swing.GroupLayout.DEFAULT_SIZE, 501, Short.MAX_VALUE)
        );
        jPanel3Layout.setVerticalGroup(
            jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel3Layout.createSequentialGroup()
                .addComponent(jToolBar2, javax.swing.GroupLayout.PREFERRED_SIZE, 25, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jToolBar5, javax.swing.GroupLayout.PREFERRED_SIZE, 25, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jScrollPane3, javax.swing.GroupLayout.DEFAULT_SIZE, 199, Short.MAX_VALUE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jToolBar4, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
        );

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                .addComponent(jPanel2, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jPanel3, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jPanel2, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addComponent(jPanel3, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
        );
    }// </editor-fold>//GEN-END:initComponents

    private class DataRepositoriesArray {

        public DataRepositoriesArray(DataRepository[] array) {
            this.array = array;
        }
        public DataRepository[] array = null;
    }
    private Map<DataBundle, DataRepositoriesArray> map = new HashMap<DataBundle, DataRepositoriesArray>();

    private class RepositoryFrameListener implements WindowListener {

        private DataBundle dataBundle;

        public RepositoryFrameListener(DataBundle dataBundle) {
            this.dataBundle = dataBundle;
        }

        public void windowOpened(WindowEvent e) {
        }

        public void windowClosing(WindowEvent e) {
        }

        public void windowClosed(WindowEvent e) {
            DataPanel.this.fillList(dataBundle);
            e.getWindow().removeWindowListener(this);
        }

        public void windowIconified(WindowEvent e) {
        }

        public void windowDeiconified(WindowEvent e) {
        }

        public void windowActivated(WindowEvent e) {
        }

        public void windowDeactivated(WindowEvent e) {
        }
    }

    private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton1ActionPerformed
        if (jList1.getSelectedValue() == null) {
            return;
        }
        int index = jList1.getSelectedIndex();
        DataBundle dataBundle = listDataBundle.get(index);

        if (dataBundle.getFrame() != null) {
            JFrame jFrame = dataBundle.getFrame();
            jFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            jFrame.addWindowListener(new RepositoryFrameListener(dataBundle));
            jFrame.setSize(jFrame.getPreferredSize());
            jFrame.setVisible(true);
        } else {
            fillList(dataBundle);
        }
    }//GEN-LAST:event_jButton1ActionPerformed

    private String prepareMsg(DataRepository group) {
        String msg = "During draw this data set, order of x axis is not provided.\n"
                + "It means that some x value is lower than some previous x value.\n"
                + "Plugin which was used to loading data, don't support ascending order of x values.\n"
                + "You should check/improve your data source.\n"
                + "This error was detected in repository with label: " + group.getLabel();
        return msg;
    }

    private boolean checkOrder(DataBlock dataBlock) {
        if (dataBlock.rows() == 1) {
            return true;
        }
        for (int fa = 1; fa < dataBlock.rows(); fa++) {
            if (dataBlock.get(fa, 0).getNumber().floatValue() < dataBlock.get(fa - 1, 0).getNumber().floatValue()) {
                return false;
            }
        }
        return true;
    }

    private void registerDisplayer(int index) {
        DataRepository dataRepository = listDataRepository.get(index);
        for (int fa = 0; fa < dataRepository.size(); fa++) {
            DataBlock dataBlock = dataRepository.get(fa);
            if (!checkOrder(dataBlock)) {
                String msg = prepareMsg(dataRepository);
                // JOptionPane.showMessageDialog(this, msg, "Error in your data source", JOptionPane.ERROR_MESSAGE);

                // return;
            }
        }
        DisplayerImpl[] displayer = new DisplayerImpl[1];
        AxisChart2DPlotManagerBundle plotManagerBundle = (AxisChart2DPlotManagerBundle) ListElementManager.get(defaultComboBoxModel.getSelectedItem());
        AxisChart2DPlotManager plotManager = plotManagerBundle.newPlotManager(axisChart);
        DisplayerTools.create(axisChart, listDataRepository.get(index), plotManager, displayer, dataRepository.getLabel(), axisIndices);
        listDisplayer.add(displayer[0]);
    }

    private void registerDisplayerGroup(int[] indices) {
        DataRepository[] array = new DataRepository[indices.length];
        for (DataRepository dataRepositoriesGroup : array) {
            for (int fa = 0; fa < dataRepositoriesGroup.size(); fa++) {
                DataBlock dataRepository = dataRepositoriesGroup.get(fa);
                if (!checkOrder(dataRepository)) {
                    String msg = prepareMsg(dataRepositoriesGroup);
                    //  JOptionPane.showMessageDialog(this, msg, "Error in your data source", JOptionPane.ERROR_MESSAGE);
                    //  return;
                }
            }
        }
        DisplayerImpl[] displayer = new DisplayerImpl[1];
        AxisChart2DPlotManagerBundle plotManagerBundle = (AxisChart2DPlotManagerBundle) ListElementManager.get(defaultComboBoxModel.getSelectedItem());
        AxisChart2DPlotManager plotManager = plotManagerBundle.newPlotManager(axisChart);
        String name = "";
        for (int fa = 0; fa < indices.length; fa++) {
            DataRepository dataRepository = listDataRepository.get(indices[fa]);
            array[fa] = dataRepository;
            name += dataRepository.getLabel() + ":";
        }

        DisplayerTools.create(axisChart, array, plotManager, displayer, name, axisIndices);
        listDisplayer.add(displayer[0]);
    }

    private void jButton2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton2ActionPerformed
        if (jList2.getSelectedIndices().length == 0) {
            return;
        }
        if (jComboBox1.getSelectedItem() == null) {
            return;
        }

        int[] indices = jList2.getSelectedIndices();

        if (!jCheckBoxAsGroup.isSelected()) {
            for (int i : indices) {
                registerDisplayer(i);
            }
        } else {
            registerDisplayerGroup(indices);
        }
        axisChart.repaintChartSurface();

    }//GEN-LAST:event_jButton2ActionPerformed

    private void jButton3ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton3ActionPerformed
        if (jList3.getSelectedIndex() == -1) {
            return;
        }

        listDisplayer.remove(jList3.getSelectedIndex());
        axisChart.repaintChartSurface();
    }//GEN-LAST:event_jButton3ActionPerformed
    private boolean isFrameOfPlotManagerActivated = false;

    private class FrameListener implements WindowListener {

        public void windowOpened(WindowEvent e) {
        }

        public void windowClosing(WindowEvent e) {
            isFrameOfPlotManagerActivated = false;
            jLabel2.setVisible(false);
            e.getWindow().removeWindowListener(this);
        }

        public void windowClosed(WindowEvent e) {
            isFrameOfPlotManagerActivated = false;
            jLabel2.setVisible(false);
            e.getWindow().removeWindowListener(this);
        }

        public void windowIconified(WindowEvent e) {
        }

        public void windowDeiconified(WindowEvent e) {
        }

        public void windowActivated(WindowEvent e) {
        }

        public void windowDeactivated(WindowEvent e) {
            isFrameOfPlotManagerActivated = false;
            e.getWindow().removeWindowListener(this);
            jLabel2.setVisible(false);
        }
    }

    private void jButton4ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton4ActionPerformed
        if (jList3.getSelectedIndex() == -1) {
            return;
        }
        if (isFrameOfPlotManagerActivated) {
            jLabel2.setVisible(true);
            return;
        }
        DisplayerImpl displayer = (DisplayerImpl) listDisplayer.get(jList3.getSelectedIndex());
        JFrame jFrame = displayer.getPlotManager().getFrame();
        if (jFrame == null) {
            return;
        }
        jFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        jFrame.setSize(jFrame.getPreferredSize());
        jFrame.addWindowListener(new FrameListener());
        jFrame.setVisible(true);
    }//GEN-LAST:event_jButton4ActionPerformed

    private void jButton5ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton5ActionPerformed
        Object item = jComboBox1.getSelectedItem();
        int j = jList3.getSelectedIndex();
        if (item == null || j == -1) {
            return;
        }
        DisplayerExtendedImpl displayer = (DisplayerExtendedImpl) listDisplayer.get(j);

        AxisChart2DPlotManagerBundle plotManagerBundle = (AxisChart2DPlotManagerBundle) ListElementManager.get(item);
        displayer.setPlotManager(plotManagerBundle.newPlotManager(axisChart));
        axisChart.repaintChartSurface();
    }//GEN-LAST:event_jButton5ActionPerformed
    private int shift = 0;
    private void jButton6ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton6ActionPerformed
        if (jCheckBox1.isSelected()) {
            for (int fa = 0; fa < listDisplayer.size(); fa++) {
                DisplayerExtendedImpl displayer = (DisplayerExtendedImpl) listDisplayer.get(fa);
                if (runnableImpl == null || !runnableImpl.playedDisplayers.contains(displayer)) {
                    displayer.setOlder();
                }
            }
        } else {
            int index = jList3.getSelectedIndex();
            if (index == -1) {
                return;
            }
            DisplayerExtendedImpl displayer = (DisplayerExtendedImpl) listDisplayer.get(index);
            if (runnableImpl == null || !runnableImpl.playedDisplayers.contains(displayer)) {
                displayer.setOlder();
            }
        }

    }//GEN-LAST:event_jButton6ActionPerformed

    private void jButton7ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton7ActionPerformed
        if (jCheckBox1.isSelected()) {
            for (int fa = 0; fa < listDisplayer.size(); fa++) {
                DisplayerExtendedImpl displayer = (DisplayerExtendedImpl) listDisplayer.get(fa);
                if (runnableImpl == null || !runnableImpl.playedDisplayers.contains(displayer)) {
                    displayer.setYounger();
                }

            }
        } else {
            int index = jList3.getSelectedIndex();
            if (index < 0) {
                return;
            }
            DisplayerExtendedImpl displayer = (DisplayerExtendedImpl) listDisplayer.get(index);
            if (runnableImpl == null || !runnableImpl.playedDisplayers.contains(displayer)) {
                displayer.setYounger();
            }
        }
    }//GEN-LAST:event_jButton7ActionPerformed

    private void jButton8ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton8ActionPerformed
        if (jCheckBox1.isSelected()) {
            for (int fa = 0; fa < listDisplayer.size(); fa++) {
                DisplayerExtendedImpl displayer = (DisplayerExtendedImpl) listDisplayer.get(fa);
                displayer.setCurrent();
            }
        } else {
            int index = jList3.getSelectedIndex();
            if (index < 0) {
                return;
            }
            DisplayerExtendedImpl displayer = (DisplayerExtendedImpl) listDisplayer.get(index);
            displayer.setCurrent();
        }
    }//GEN-LAST:event_jButton8ActionPerformed

    private class RunnableImpl implements Runnable {

        public List<DisplayerExtendedImpl> playedDisplayers = new ArrayList<DisplayerExtendedImpl>();
        private boolean stop = false;
        private int seconds = 0;
        private final Object sync = new Float(1);

        public void stop() {
            this.stop = true;
        }

        public RunnableImpl() {
            playedDisplayers = Collections.synchronizedList(playedDisplayers);
        }

        public void run() {
            seconds = (int) ((Float) jSpinner1.getValue() * 1000.f);
            for (DisplayerExtendedImpl displayer : playedDisplayers) {
                displayer.setTheOldest();
            }
            axisChart.repaintChartSurface();

            while (stop == false) {
                try {
                    for (DisplayerExtendedImpl displayer : playedDisplayers) {
                        if (!displayer.setYounger()) {
                            displayer.setTheOldest();
                        }
                    }
                    axisChart.repaintChartSurface();
                    synchronized (sync) {
                        sync.wait(seconds);
                    }
                } catch (InterruptedException ex) {
                    Logger.getLogger(DataPanel.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
            playedDisplayers.removeAll(playedDisplayers);
            jSpinner1.setEnabled(true);
        }
    }
    private RunnableImpl runnableImpl = null;
    private RunnableImpl runnableImpl1 = null;
    private void jButton9ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton9ActionPerformed
        if (jList3.getSelectedIndices().length == 0) {
            JOptionPane.showMessageDialog(this, "Select repositoriy from list.");
            return;
        }

        if (runnableImpl != null) {
            return;
        }
        jSpinner1.setEnabled(false);
        if (runnableImpl == null) {
            runnableImpl = new RunnableImpl();
            runnableImpl1 = runnableImpl;
        } else {
            runnableImpl = runnableImpl1;
        }
        if (jCheckBox1.isSelected()) {
            for (int fa = 0; fa < listDisplayer.size(); fa++) {
                DisplayerExtendedImpl displayer = (DisplayerExtendedImpl) listDisplayer.get(fa);
                if (displayer.getDataRepository().getHistory() != null) {
                    runnableImpl.playedDisplayers.add(displayer);
                }
            }
        } else {
            int index = jList3.getSelectedIndex();
            if (index < 0) {
                return;
            }
            DisplayerExtendedImpl displayer = (DisplayerExtendedImpl) listDisplayer.get(index);
            if (displayer.getDataRepository().getHistory() != null) {
                runnableImpl.playedDisplayers.add(displayer);
            }
        }
        if (runnableImpl.playedDisplayers.size() > 0) {
            Thread thread = new Thread(runnableImpl);
            thread.start();
        }
    }//GEN-LAST:event_jButton9ActionPerformed

    private void jButton10ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton10ActionPerformed
        if (runnableImpl != null) {
            runnableImpl.stop();
            runnableImpl = null;
        }
    }//GEN-LAST:event_jButton10ActionPerformed
    private TableFrame tableFrame = new TableFrame();

    private class PopupMenu extends JPopupMenu {

        private JMenuItem menuItem = new JMenuItem("Show data");

        public PopupMenu() {
            super();
            this.add(menuItem);
            menuItem.addActionListener(new ActionListener() {

                public void actionPerformed(ActionEvent e) {
                    int index = jList2.getSelectedIndex();
                    if (index == -1) {
                        return;
                    }
                    tableFrame.setDataRepository(axisChart.getListDataRepository().get(index));
                    tableFrame.setVisible(true);
                }
            });
        }
    }
    private PopupMenu popupMenu = new PopupMenu();

    private void jList2MouseReleased(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_jList2MouseReleased
        if (evt.getButton() == java.awt.event.MouseEvent.BUTTON3) {
            popupMenu.show(DataPanel.this.jList2, evt.getX(), evt.getY());
        }
    }//GEN-LAST:event_jList2MouseReleased

    private void jComboBox1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jComboBox1ActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_jComboBox1ActionPerformed
    private AxisIndices axisIndices = new AxisIndices();

    private AxisIndices getAxisIndicesOfSelection() {
        AxisIndices indices = axisIndices;
        if (jList3.getSelectedIndex() > -1) {
            Displayer displayer = this.listDisplayer.get(jList3.getSelectedIndex());
            DisplayerExtendedImpl dei = (DisplayerExtendedImpl) displayer;
            indices = dei.getAxisIndices();
            indices.giveTo(axisIndices);
            return indices;
        }
        return axisIndices;
    }
    private void jSpinner2StateChanged(javax.swing.event.ChangeEvent evt) {//GEN-FIRST:event_jSpinner2StateChanged
        AxisIndices indices = getAxisIndicesOfSelection();
        Number n = (Number) jSpinner2.getValue();
        indices.setIndexX(n.intValue());
        axisChart.repaintChartSurface();
    }//GEN-LAST:event_jSpinner2StateChanged

    private void jSpinner3StateChanged(javax.swing.event.ChangeEvent evt) {//GEN-FIRST:event_jSpinner3StateChanged
        AxisIndices indices = getAxisIndicesOfSelection();
        Number n = (Number) jSpinner3.getValue();
        indices.setIndexY(n.intValue());
        axisChart.repaintChartSurface();
    }//GEN-LAST:event_jSpinner3StateChanged

    private void jComboBox2ItemStateChanged(java.awt.event.ItemEvent evt) {//GEN-FIRST:event_jComboBox2ItemStateChanged
        AxisIndices indices = getAxisIndicesOfSelection();
        String n = jComboBox2.getSelectedItem().toString();
        if (n.equals("From begin")) {
            indices.setSideX(AxisIndices.FromBegin);
        } else if (n.equals("From end")) {
            indices.setSideX(AxisIndices.FromEnd);
        }
        axisChart.repaintChartSurface();
    }//GEN-LAST:event_jComboBox2ItemStateChanged

    private void jComboBox2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jComboBox2ActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_jComboBox2ActionPerformed

    private void jComboBox3ItemStateChanged(java.awt.event.ItemEvent evt) {//GEN-FIRST:event_jComboBox3ItemStateChanged
        AxisIndices indices = axisIndices;
        if (jList3.getSelectedIndex() > -1) {
            Displayer displayer = this.listDisplayer.get(jList3.getSelectedIndex());
            DisplayerExtendedImpl dei = (DisplayerExtendedImpl) displayer;
            indices = dei.getAxisIndices();
        }
        String n = jComboBox3.getSelectedItem().toString();
        if (n.equals("From begin")) {
            indices.setSideY(AxisIndices.FromBegin);
        } else if (n.equals("From end")) {
            indices.setSideY(AxisIndices.FromEnd);
        }
        axisChart.repaintChartSurface();
    }//GEN-LAST:event_jComboBox3ItemStateChanged

    private void jList3ValueChanged(javax.swing.event.ListSelectionEvent evt) {//GEN-FIRST:event_jList3ValueChanged
        AxisIndices axisIndices = getAxisIndicesOfSelection();
        jSpinner2.setValue(axisIndices.getRawXIndex());
        jComboBox2.setSelectedIndex(axisIndices.getSideX());
        jSpinner3.setValue(axisIndices.getRawYIndex());
        jComboBox3.setSelectedIndex(axisIndices.getSideY());
    }//GEN-LAST:event_jList3ValueChanged
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton jButton1;
    private javax.swing.JButton jButton10;
    private javax.swing.JButton jButton2;
    private javax.swing.JButton jButton3;
    private javax.swing.JButton jButton4;
    private javax.swing.JButton jButton5;
    private javax.swing.JButton jButton6;
    private javax.swing.JButton jButton7;
    private javax.swing.JButton jButton8;
    private javax.swing.JButton jButton9;
    private javax.swing.JCheckBox jCheckBox1;
    private javax.swing.JCheckBox jCheckBoxAsGroup;
    private javax.swing.JComboBox jComboBox1;
    private javax.swing.JComboBox jComboBox2;
    private javax.swing.JComboBox jComboBox3;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JLabel jLabel3;
    private javax.swing.JLabel jLabel4;
    private javax.swing.JList jList1;
    private javax.swing.JList jList2;
    private javax.swing.JList jList3;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JPanel jPanel2;
    private javax.swing.JPanel jPanel3;
    private javax.swing.JPanel jPanel4;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JScrollPane jScrollPane2;
    private javax.swing.JScrollPane jScrollPane3;
    private javax.swing.JToolBar.Separator jSeparator1;
    private javax.swing.JToolBar.Separator jSeparator2;
    private javax.swing.JToolBar.Separator jSeparator3;
    private javax.swing.JToolBar.Separator jSeparator4;
    private javax.swing.JSpinner jSpinner1;
    private javax.swing.JSpinner jSpinner2;
    private javax.swing.JSpinner jSpinner3;
    private javax.swing.JSplitPane jSplitPane1;
    private javax.swing.JToolBar jToolBar1;
    private javax.swing.JToolBar jToolBar2;
    private javax.swing.JToolBar jToolBar3;
    private javax.swing.JToolBar jToolBar4;
    private javax.swing.JToolBar jToolBar5;
    // End of variables declaration//GEN-END:variables
}
