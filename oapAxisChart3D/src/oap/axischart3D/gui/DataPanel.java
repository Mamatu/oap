package ogla.axischart3D.gui;

import ogla.axischart.gui.TableFrame;
import ogla.data.DataBundle;
import ogla.data.DataRepository;
import ogla.data.DataBlock;
import ogla.axischart.Displayer;
import ogla.axischart.lists.ListDataBundle;
import ogla.axischart.lists.ListDataRepository;
import ogla.axischart.lists.ListDisplayer;
import ogla.axischart.lists.ListListener;
import ogla.axischart.lists.ListsContainer;
import ogla.core.util.ListElementManager;
import ogla.axischart3D.AxisChart3D;
import ogla.axischart3D.lists.ListAxisChart3DPlotManagerBundle;
import ogla.axischart3D.lists.ListDataRepositoryWithPrimitives;
import ogla.axischart3D.plugin.plotmanager.AxisChart3DPlotManager;
import ogla.axischart3D.plugin.plotmanager.AxisChart3DPlotManagerBundle;
import ogla.axischart3D.util.DisplayerImpl;
import ogla.axischart3D.util.DisplayerTools;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.swing.DefaultComboBoxModel;
import javax.swing.DefaultListModel;
import javax.swing.JFrame;
import javax.swing.JMenuItem;
import javax.swing.JPopupMenu;

public final class DataPanel extends javax.swing.JPanel {

    protected AxisChart3D axisChart3D;
    protected ListsContainer listsContainer;
    protected DefaultComboBoxModel defaultComboBoxModel = new DefaultComboBoxModel();
    protected DefaultComboBoxModel defaultComboBoxModel1 = new DefaultComboBoxModel();
    protected DefaultListModel defaultListModel3 = new DefaultListModel();
    protected DefaultListModel defaultListModel1 = new DefaultListModel();
    protected DefaultListModel defaultListModel2 = new DefaultListModel();
    protected DefaultListModel defaultListModel4 = new DefaultListModel();
    private ListDisplayer listDisplayer = null;
    private ListDataRepository listDataRepository = null;
    private ListDataBundle listDataBundle = null;
    private ListAxisChart3DPlotManagerBundle listAxisChart3DPlotManagerBundle = null;
    private ListDataRepositoryWithPrimitives listDataRepositoryWithPrimitives = null;

    /** Creates new form DataPanel */
    public DataPanel(AxisChart3D axisChart, ListsContainer listsContainer) {
        initComponents();
        jLabel2.setVisible(false);
        this.axisChart3D = axisChart;
        this.listsContainer = listsContainer;
        jList1.setModel(defaultListModel1);
        jList2.setModel(defaultListModel2);
        jList3.setModel(defaultListModel3);
        jComboBox1.setModel(defaultComboBoxModel1);
        this.listDisplayer = listsContainer.get(ListDisplayer.class);

        this.listDataBundle = listsContainer.get(ListDataBundle.class);
        this.listAxisChart3DPlotManagerBundle = listsContainer.get(ListAxisChart3DPlotManagerBundle.class);

        this.listAxisChart3DPlotManagerBundle.addListListener(new ListListener<AxisChart3DPlotManagerBundle>() {

            public void isAdded(AxisChart3DPlotManagerBundle t) {
                defaultComboBoxModel1.addElement(ListElementManager.register(t, t.getLabel()));
            }

            public void isRemoved(AxisChart3DPlotManagerBundle t) {
                defaultComboBoxModel1.removeElementAt(ListElementManager.getIndex(t, defaultComboBoxModel1));
            }
        });

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

        this.listDataRepositoryWithPrimitives = listsContainer.get(ListDataRepositoryWithPrimitives.class);
        this.listDataRepositoryWithPrimitives.addListListener(new ListListener<DataRepository>() {

            public void isAdded(DataRepository t) {
                defaultListModel4.addElement(ListElementManager.register(t));
            }

            public void isRemoved(DataRepository t) {
                defaultListModel4.removeElementAt(ListElementManager.getIndex(t, defaultListModel4));
            }
        });


    }
    private Map<DataBundle, List<DataRepository>> bundles = new HashMap<DataBundle, List<DataRepository>>();

    public void fillList(DataBundle dataBundle) {
        if (bundles.containsKey(dataBundle)) {
            List<DataRepository> dataRepositorys = bundles.get(dataBundle);
            for (int fa = 0; fa < dataRepositorys.size(); fa++) {
                DataRepository dataRepository = dataRepositorys.get(fa);
                for (int fb = 0; fb < dataRepository.size(); fb++) {
                    DataBlock dataBlock = dataRepository.get(fb);
                    axisChart3D.disattachPrimitives(dataBlock);
                }
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

        jSeparator1 = new javax.swing.JToolBar.Separator();
        jToolBar4 = new javax.swing.JToolBar();
        jButton6 = new javax.swing.JButton();
        jButton8 = new javax.swing.JButton();
        jButton7 = new javax.swing.JButton();
        jButton9 = new javax.swing.JButton();
        jButton10 = new javax.swing.JButton();
        jSeparator2 = new javax.swing.JToolBar.Separator();
        jSpinner1 = new javax.swing.JSpinner();
        jSeparator3 = new javax.swing.JToolBar.Separator();
        jCheckBox1 = new javax.swing.JCheckBox();
        jDesktopPane1 = new javax.swing.JDesktopPane();
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
        jToolBar5 = new javax.swing.JToolBar();
        jButton11 = new javax.swing.JButton();
        jComboBox1 = new javax.swing.JComboBox();
        jButton5 = new javax.swing.JButton();
        jButton3 = new javax.swing.JButton();
        jLabel2 = new javax.swing.JLabel();

        jToolBar4.setFloatable(false);
        jToolBar4.setRollover(true);

        jButton6.setText("<");
        jButton6.setFocusable(false);
        jButton6.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jButton6.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        jToolBar4.add(jButton6);

        jButton8.setText("Return");
        jButton8.setToolTipText("Return to the most current repository");
        jButton8.setFocusable(false);
        jButton8.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jButton8.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        jToolBar4.add(jButton8);

        jButton7.setText(">");
        jButton7.setFocusable(false);
        jButton7.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jButton7.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        jToolBar4.add(jButton7);

        jButton9.setText(">>");
        jButton9.setToolTipText("Play");
        jButton9.setFocusable(false);
        jButton9.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jButton9.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        jToolBar4.add(jButton9);

        jButton10.setText("||");
        jButton10.setFocusable(false);
        jButton10.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jButton10.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
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
            .addComponent(jScrollPane1, javax.swing.GroupLayout.DEFAULT_SIZE, 72, Short.MAX_VALUE)
            .addComponent(jToolBar3, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
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
            .addComponent(jScrollPane2, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, 143, Short.MAX_VALUE)
            .addComponent(jToolBar1, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
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
        jScrollPane3.setViewportView(jList3);

        jToolBar5.setFloatable(false);
        jToolBar5.setRollover(true);

        jButton11.setText("Change");
        jButton11.setFocusable(false);
        jButton11.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jButton11.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        jButton11.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton11ActionPerformed(evt);
            }
        });
        jToolBar5.add(jButton11);

        jComboBox1.addItemListener(new java.awt.event.ItemListener() {
            public void itemStateChanged(java.awt.event.ItemEvent evt) {
                jComboBox1ItemStateChanged(evt);
            }
        });
        jToolBar5.add(jComboBox1);

        jButton5.setText("Configure");
        jButton5.setFocusable(false);
        jButton5.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jButton5.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        jButton5.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton5ActionPerformed(evt);
            }
        });
        jToolBar5.add(jButton5);

        jButton3.setText("Hide");
        jButton3.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton3ActionPerformed(evt);
            }
        });
        jToolBar5.add(jButton3);

        jLabel2.setForeground(new java.awt.Color(231, 28, 28));
        jLabel2.setText("Another frame is activated.");
        jToolBar5.add(jLabel2);

        javax.swing.GroupLayout jPanel3Layout = new javax.swing.GroupLayout(jPanel3);
        jPanel3.setLayout(jPanel3Layout);
        jPanel3Layout.setHorizontalGroup(
            jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jToolBar5, javax.swing.GroupLayout.DEFAULT_SIZE, 480, Short.MAX_VALUE)
            .addComponent(jScrollPane3, javax.swing.GroupLayout.DEFAULT_SIZE, 480, Short.MAX_VALUE)
        );
        jPanel3Layout.setVerticalGroup(
            jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel3Layout.createSequentialGroup()
                .addComponent(jToolBar5, javax.swing.GroupLayout.PREFERRED_SIZE, 25, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jScrollPane3, javax.swing.GroupLayout.DEFAULT_SIZE, 282, Short.MAX_VALUE))
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
        DataBundle bundleData = listDataBundle.get(index);
        JFrame jFrame = bundleData.getFrame();

        if (jFrame != null) {
            jFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            jFrame.addWindowListener(new RepositoryFrameListener(bundleData));
            jFrame.setSize(jFrame.getPreferredSize());
            jFrame.setVisible(true);
        } else {
            fillList(bundleData);
        }
    }//GEN-LAST:event_jButton1ActionPerformed

    private boolean checkOrder(DataBlock dataRepository) {
        if (dataRepository.rows() == 1) {
            return true;
        }
        for (int fa = 1; fa < dataRepository.rows(); fa++) {
            if (dataRepository.get(fa, 0).getNumber().floatValue() < dataRepository.get(fa - 1, 0).getNumber().floatValue()) {
                return false;
            }
        }
        return true;
    }

    private void registerDisplayer(int index) {
        DataRepository dataRepository = listDataRepository.get(index);
        DisplayerImpl[] displayer = new DisplayerImpl[1];

        int i = ListElementManager.getIndex(defaultComboBoxModel1.getSelectedItem(), defaultComboBoxModel1);
        AxisChart3DPlotManagerBundle axisChart3DPlotManagerBundle =
                listAxisChart3DPlotManagerBundle.get(i);
        DisplayerTools.create(axisChart3D, listDataRepository.get(index), displayer,
                dataRepository.getLabel(), axisChart3DPlotManagerBundle.newPlotManager());
        listDisplayer.add(displayer[0]);
    }

    private void registerDisplayerGroup(int[] indices) {
        DataRepository[] array = new DataRepository[indices.length];
        DisplayerImpl[] displayer = new DisplayerImpl[1];
        String name = "";
        for (int fa = 0; fa < indices.length; fa++) {
            DataRepository dataRepository = listDataRepository.get(indices[fa]);
            array[fa] = dataRepository;
            name += dataRepository.getLabel() + ":";
        }
        int index = ListElementManager.getIndex(defaultComboBoxModel1.getSelectedItem(), defaultComboBoxModel1);
        AxisChart3DPlotManagerBundle axisChart3DPlotManagerBundle =
                listAxisChart3DPlotManagerBundle.get(index);
        DisplayerTools.create(axisChart3D, array, displayer, name, axisChart3DPlotManagerBundle.newPlotManager());
        listDisplayer.add(displayer[0]);
    }

    private void jButton2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton2ActionPerformed
        if (jList2.getSelectedIndices().length == 0) {
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


    }//GEN-LAST:event_jButton2ActionPerformed

    private void jButton3ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton3ActionPerformed
        if (jList3.getSelectedIndex() == -1) {
            return;
        }
        final int index = jList3.getSelectedIndex();
        listDisplayer.remove(index);
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
    private TableFrame showDataFrame = new TableFrame();

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
                    showDataFrame.setDataRepository(axisChart3D.getListDataRepository().get(index));
                    showDataFrame.setVisible(true);
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

    private class WindowListenerImpl implements WindowListener {

        public void windowOpened(WindowEvent e) {
        }

        public void windowClosing(WindowEvent e) {
        }

        public void windowClosed(WindowEvent e) {
            preparePrimitives();
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
    private WindowListenerImpl windowListenerImpl = new WindowListenerImpl();

    private void jButton5ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton5ActionPerformed
        int index = jList3.getSelectedIndex();
        if (index == -1) {
            return;
        }
        DataRepository dataRepository = listDataRepository.get(index);
        AxisChart3DPlotManager axisChart3DPlotManager = this.axisChart3D.getAxisChart3DPlotManagerOf(dataRepository);
        if (axisChart3DPlotManager == null) {
            return;
        }
        JFrame jFrame = axisChart3DPlotManager.getFrame();
        if (jFrame == null) {
            return;
        }
        jFrame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        jFrame.setSize(jFrame.getPreferredSize());
        jFrame.setVisible(true);
        jFrame.addWindowListener(windowListenerImpl);

    }//GEN-LAST:event_jButton5ActionPerformed

    private void preparePrimitives(AxisChart3DPlotManagerBundle axisChart3DPlotManagerBundle, DataRepository dataRepository) {
        axisChart3D.putNewPrimitives(dataRepository, axisChart3DPlotManagerBundle.newPlotManager());
    }

    private void preparePrimitives() {
        int index = jList3.getSelectedIndex();
        if (index == -1) {
            return;
        }
        int i = jComboBox1.getSelectedIndex();
        AxisChart3DPlotManagerBundle axisChart3DPlotManagerBundle = listAxisChart3DPlotManagerBundle.get(i);
        DisplayerImpl displayerImpl = (DisplayerImpl) ListElementManager.get(jList3.getSelectedValue());
        for (int fa = 0; fa < displayerImpl.getNumberOfDataRepositories(); fa++) {
            DataRepository dataRepository = displayerImpl.getDataRepository(fa);
            preparePrimitives(axisChart3DPlotManagerBundle, dataRepository);
        }
    }

    private void jButton11ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton11ActionPerformed
        preparePrimitives();
    }//GEN-LAST:event_jButton11ActionPerformed

    private void jComboBox1ItemStateChanged(java.awt.event.ItemEvent evt) {//GEN-FIRST:event_jComboBox1ItemStateChanged
        int i = jComboBox1.getSelectedIndex();
        AxisChart3DPlotManagerBundle axisChart3DPlotManagerBundle = listAxisChart3DPlotManagerBundle.get(i);
        axisChart3D.setAxisChart3DPlotManagerBundle(axisChart3DPlotManagerBundle);
    }//GEN-LAST:event_jComboBox1ItemStateChanged
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton jButton1;
    private javax.swing.JButton jButton10;
    private javax.swing.JButton jButton11;
    private javax.swing.JButton jButton2;
    private javax.swing.JButton jButton3;
    private javax.swing.JButton jButton5;
    private javax.swing.JButton jButton6;
    private javax.swing.JButton jButton7;
    private javax.swing.JButton jButton8;
    private javax.swing.JButton jButton9;
    private javax.swing.JCheckBox jCheckBox1;
    private javax.swing.JCheckBox jCheckBoxAsGroup;
    private javax.swing.JComboBox jComboBox1;
    private javax.swing.JDesktopPane jDesktopPane1;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
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
    private javax.swing.JSpinner jSpinner1;
    private javax.swing.JSplitPane jSplitPane1;
    private javax.swing.JToolBar jToolBar1;
    private javax.swing.JToolBar jToolBar3;
    private javax.swing.JToolBar jToolBar4;
    private javax.swing.JToolBar jToolBar5;
    // End of variables declaration//GEN-END:variables
}
