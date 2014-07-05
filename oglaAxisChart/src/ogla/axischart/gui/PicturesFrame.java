package ogla.axischart.gui;

import ogla.axischart.plugin.exporter.BundleImageExporter;
import ogla.axischart.plugin.exporter.ImageExporter;
import ogla.axischart.plugin.image.ImageComponent;
import ogla.axischart.lists.ListImageExporterBundle;
import ogla.axischart.lists.ListInstalledImageComponent;
import ogla.axischart.lists.ListListener;
import ogla.axischart.lists.ListsContainer;
import ogla.axischart.AxisChart;
import java.awt.event.ActionEvent;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.IOException;
import java.util.Iterator;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import javax.imageio.ImageWriter;
import javax.imageio.stream.ImageOutputStream;
import javax.swing.JFileChooser;
import ogla.axischart.util.ImageList;
import ogla.core.util.ListElementManager;
import java.awt.image.BufferedImage;
import javax.swing.DefaultListModel;
import ogla.chart.ChartSurface;
import ogla.chart.DrawingInfo;
import java.awt.Image;
import java.awt.event.ActionListener;
import java.awt.event.WindowListener;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JFrame;
import javax.swing.JMenuItem;
import javax.swing.JPopupMenu;
import javax.swing.filechooser.FileFilter;

/**
 *
 * @author marcin
 */
public class PicturesFrame extends javax.swing.JFrame {

    DefaultListModel globalImageListModel = null;
    DefaultListModel imageListModel = null;
    private Group globalGroup = null;
    private AxisChart axisChart = null;
    private ChartSurface chartSurface;
    private DefaultListModel listModelImageComponent = new DefaultListModel();
    private ListsContainer listsContainer;

    private class Popup extends JPopupMenu {

        private JMenuItem item1 = new JMenuItem("Current chart to image");
        private JMenuItem item2 = new JMenuItem("Selected image + selected plugin");
        private JMenuItem item3 = new JMenuItem("Current chart + selected plugin");

        public Popup() {
            super();
            item1.addActionListener(new ActionListener() {

                public void actionPerformed(ActionEvent e) {
                    currentChartToImage();
                }
            });

            item2.addActionListener(new ActionListener() {

                public void actionPerformed(ActionEvent e) {
                    selectedPluginsAndSelectedImage();
                }
            });

            item3.addActionListener(new ActionListener() {

                public void actionPerformed(ActionEvent e) {
                    selectedPluginsPlusCurrentChart();
                }
            });
            this.add(item1);
            this.add(item2);
            this.add(item3);
        }
    }

    private class Tuple {

        public Tuple(ImageComponent imageComponent, String label) {
            this.label = label;
            this.imageComponent = imageComponent;
        }
        public ImageComponent imageComponent;
        public String label;

        @Override
        public String toString() {
            return label;
        }
    }
    private List<Tuple> tuplesToRemoved = new ArrayList<Tuple>();

    private class ListListener1 implements ListListener<ImageComponent> {

        public void isAdded(ImageComponent t) {
            listModelImageComponent.addElement(ListElementManager.register(t, t.getBundle().getLabel()));
        }

        public void isRemoved(ImageComponent t) {
            listModelImageComponent.removeElementAt(ListElementManager.getIndex(t, listModelImageComponent));
        }
    }

    public PicturesFrame(DefaultListModel imageListModel, AxisChart axisChart, ListsContainer listsContainer) {
        initComponents();
        jList1.setCellRenderer(new ImageList());
        jList1.setModel(imageListModel);
        jFileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        this.globalImageListModel = imageListModel;
        this.imageListModel = this.globalImageListModel;
        globalGroup = new Group("All", this.globalImageListModel);
        this.axisChart = axisChart;
        this.chartSurface = axisChart.getChartSurface();
        this.listsContainer = listsContainer;
        groupsModel.addElement(globalGroup);
        jComboBox2.setModel(groupsModel);


        jList2.setModel(listModelImageComponent);
        ListInstalledImageComponent listInstalledImageComponent = listsContainer.get(ListInstalledImageComponent.class);
        listInstalledImageComponent.addListListener(new ListListener1());

        ListImageExporterBundle listImageExporterBundle = listsContainer.get(ListImageExporterBundle.class);
        listImageExporterBundle.addListListener(new ExporterListener());
    }

    /** This method is called from within the constructor to
     * initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is
     * always regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jScrollPane1 = new javax.swing.JScrollPane();
        jList1 = new javax.swing.JList();
        jScrollPane2 = new javax.swing.JScrollPane();
        jList2 = new javax.swing.JList();
        jToolBar1 = new javax.swing.JToolBar();
        jButton2 = new javax.swing.JButton();
        jSeparator3 = new javax.swing.JToolBar.Separator();
        jButton5 = new javax.swing.JButton();
        jButton3 = new javax.swing.JButton();
        jSeparator2 = new javax.swing.JToolBar.Separator();
        jComboBox2 = new javax.swing.JComboBox();
        jButton4 = new javax.swing.JButton();
        jSeparator1 = new javax.swing.JToolBar.Separator();
        jButton1 = new javax.swing.JButton();

        jList1.setCursor(new java.awt.Cursor(java.awt.Cursor.DEFAULT_CURSOR));
        jList1.setLayoutOrientation(javax.swing.JList.HORIZONTAL_WRAP);
        jScrollPane1.setViewportView(jList1);

        jScrollPane2.setViewportView(jList2);

        jToolBar1.setFloatable(false);
        jToolBar1.setRollover(true);

        jButton2.setText("Save");
        jButton2.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton2ActionPerformed(evt);
            }
        });
        jToolBar1.add(jButton2);
        jToolBar1.add(jSeparator3);

        jButton5.setText("Show");
        jButton5.setFocusable(false);
        jButton5.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jButton5.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        jButton5.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton5ActionPerformed(evt);
            }
        });
        jToolBar1.add(jButton5);

        jButton3.setText("Remove");
        jButton3.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton3ActionPerformed(evt);
            }
        });
        jToolBar1.add(jButton3);
        jToolBar1.add(jSeparator2);

        jComboBox2.addItemListener(new java.awt.event.ItemListener() {
            public void itemStateChanged(java.awt.event.ItemEvent evt) {
                jComboBox2ItemStateChanged(evt);
            }
        });
        jToolBar1.add(jComboBox2);

        jButton4.setText("Groups");
        jButton4.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton4ActionPerformed(evt);
            }
        });
        jToolBar1.add(jButton4);
        jToolBar1.add(jSeparator1);

        jButton1.setText("Image");
        jButton1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton1ActionPerformed(evt);
            }
        });
        jToolBar1.add(jButton1);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jToolBar1, javax.swing.GroupLayout.DEFAULT_SIZE, 762, Short.MAX_VALUE)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                .addComponent(jScrollPane1, javax.swing.GroupLayout.DEFAULT_SIZE, 597, Short.MAX_VALUE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jScrollPane2, javax.swing.GroupLayout.PREFERRED_SIZE, 159, javax.swing.GroupLayout.PREFERRED_SIZE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(jToolBar1, javax.swing.GroupLayout.PREFERRED_SIZE, 25, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(jScrollPane1, javax.swing.GroupLayout.DEFAULT_SIZE, 458, Short.MAX_VALUE)
                    .addComponent(jScrollPane2, javax.swing.GroupLayout.DEFAULT_SIZE, 458, Short.MAX_VALUE)))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents
    private JFileChooser jFileChooser = new JFileChooser();

    private void currentChartToImage() {
        BufferedImage bufferedImage = new BufferedImage(axisChart.getChartSurface().getWidth(),
                axisChart.getChartSurface().getHeight(), BufferedImage.TYPE_INT_ARGB);
        axisChart.getChartSurface().setDrawingType(DrawingInfo.DrawingOnImage);
        axisChart.getChartSurface().paint(bufferedImage.getGraphics());
        ImageList.Element e = new ImageList.Element("CurrentChart", bufferedImage);
        addImage(e);

    }

    private void addImage(ImageList.Element e) {
        this.globalImageListModel.add(0, e);
        if (this.imageListModel != this.globalImageListModel) {
            this.imageListModel.add(0, e);
        }
    }

    private class WindowListenerImpl implements WindowListener {

        public ImageComponent imageComponent;
        public JFrame frame = null;

        public void windowOpened(WindowEvent e) {
        }

        public void windowClosing(WindowEvent e) {
        }

        public void windowClosed(WindowEvent e) {
            imageComponent.process();
            BufferedImage newBufferedImage = imageComponent.getBufferedImage();
            ImageList.Element element = new ImageList.Element("", newBufferedImage);
            addImage(element);
            frame.removeWindowListener(this);
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

    private void selectedPluginsPlusCurrentChart() {
        int index = jList2.getSelectedIndex();
        if (index == -1) {
            return;
        }
        axisChart.getChartSurface().setDrawingType(DrawingInfo.DrawingOnImage);
        BufferedImage bufferedImage = new BufferedImage(chartSurface.getWidth(),
                chartSurface.getHeight(), BufferedImage.TYPE_INT_RGB);
        chartSurface.paint(bufferedImage.getGraphics());
        ListInstalledImageComponent listInstalledImageComponent = listsContainer.get(ListInstalledImageComponent.class);
        ImageComponent imageComponent = listInstalledImageComponent.get(index);
        imageComponent.setBufferedImage(bufferedImage);
        imageComponent.attachChart(axisChart);
        if (imageComponent.getFrame() != null) {
            windowListenerImpl.imageComponent = imageComponent;
            JFrame jFrame = imageComponent.getFrame();
            windowListenerImpl.frame = jFrame;
            jFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            jFrame.setSize(jFrame.getPreferredSize());
            jFrame.addWindowListener(windowListenerImpl);
            jFrame.setVisible(true);
        } else {
            imageComponent.process();
            BufferedImage newBufferedImage = imageComponent.getBufferedImage();
            ImageList.Element e = new ImageList.Element("", newBufferedImage);
            addImage(e);
        }

    }

    private void selectedPluginsAndSelectedImage() {
        int index = jList2.getSelectedIndex();
        int index1 = jList1.getSelectedIndex();
        if (index == -1 || index1 == -1) {
            return;
        }
        axisChart.getChartSurface().setDrawingType(DrawingInfo.DrawingOnImage);
        ImageList.Element e = (ImageList.Element) imageListModel.getElementAt(index1);
        BufferedImage bufferedImage = e.bufferedImage;
        chartSurface.paint(bufferedImage.getGraphics());
        ListInstalledImageComponent listInstalledImageComponent = listsContainer.get(ListInstalledImageComponent.class);
        ImageComponent imageComponent = listInstalledImageComponent.get(index);
        imageComponent.setBufferedImage(bufferedImage);
        imageComponent.attachChart(axisChart);
        if (imageComponent.getFrame() != null) {
            windowListenerImpl.imageComponent = imageComponent;
            JFrame jFrame = imageComponent.getFrame();
            windowListenerImpl.frame = jFrame;
            jFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            jFrame.setSize(jFrame.getPreferredSize());
            jFrame.addWindowListener(windowListenerImpl);
            jFrame.setVisible(true);
        } else {
            imageComponent.process();
            BufferedImage newBufferedImage = imageComponent.getBufferedImage();
            ImageList.Element ed = new ImageList.Element("", newBufferedImage);
            addImage(ed);
        }

    }

    private class FileFilterImpl extends FileFilter {

        public ImageExporter exporter;

        public FileFilterImpl(ImageExporter exporter) {
            this.exporter = exporter;
        }

        @Override
        public boolean accept(File file) {
            return file.isDirectory() || exporter.accept(file.getName());
        }

        @Override
        public String getDescription() {
            return exporter.getDescription();
        }

        public boolean hasImageExporter(ImageExporter imageExporter) {
            return imageExporter == this.exporter;
        }
    }
    private Map<BundleImageExporter, List<ImageExporter>> bundleExporter = new HashMap<BundleImageExporter, List<ImageExporter>>();

    private class ExporterListener implements ListListener<BundleImageExporter> {

        public void isAdded(BundleImageExporter t) {
            ImageExporter exporter = t.newImageExporter();
            jFileChooser.setFileFilter(new FileFilterImpl(exporter));
            if (bundleExporter.containsKey(t)) {
                bundleExporter.get(t).add(exporter);
            } else {
                List<ImageExporter> exporters = new ArrayList<ImageExporter>();
                exporters.add(exporter);
                bundleExporter.put(t, exporters);
            }
        }

        public void isRemoved(BundleImageExporter t) {
            if (bundleExporter.containsKey(t)) {
                bundleExporter.get(t);
            }
        }
    }

    private void nextPartOfSaving(FileFilterImpl ff) {
        BufferedImage bi = ff.exporter.getImage();
        ImageWriter writer = ff.exporter.getImageWriter();
        ImageOutputStream ios = null;

        File f = jFileChooser.getSelectedFile();

        if (!f.getName().toLowerCase().endsWith(ff.exporter.getSuffix())) {
            File temp = new File(f.getAbsolutePath() + "." + ff.exporter.getSuffix());
            f = temp;
        }

        if (f == null) {
            return;
        }
        try {
            ios = ImageIO.createImageOutputStream(f);
            writer.setOutput(ios);
            writer.write(bi);

        } catch (IOException ex) {
            Logger.getLogger(PicturesFrame.class.getName()).log(Level.SEVERE, null, ex);
        } finally {
            if (ios != null) {
                try {
                    ios.close();
                } catch (IOException ex) {
                    Logger.getLogger(PicturesFrame.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        }
    }

    class FrameListener implements WindowListener {

        private FileFilterImpl ff;
        private JFrame jFrame;

        public FrameListener(JFrame jFrame, FileFilterImpl ff) {
            this.ff = ff;
            this.jFrame = jFrame;
        }

        public void windowOpened(WindowEvent e) {
        }

        public void windowClosing(WindowEvent e) {
        }

        public void windowClosed(WindowEvent e) {
        }

        public void windowIconified(WindowEvent e) {
        }

        public void windowDeiconified(WindowEvent e) {
        }

        public void windowActivated(WindowEvent e) {
        }

        public void windowDeactivated(WindowEvent e) {
            nextPartOfSaving(ff);
            this.jFrame.removeWindowListener(this);
        }
    }

    private void jButton2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton2ActionPerformed
        int returnVal = jFileChooser.showSaveDialog(this);

        if (returnVal == JFileChooser.APPROVE_OPTION) {
            FileFilterImpl ff = (FileFilterImpl) jFileChooser.getFileFilter();
            ImageList.Element element = (ImageList.Element) jList1.getSelectedValue();
            if (element == null) {
                return;
            }
            jFileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);

            BufferedImage bufferedImage = element.bufferedImage;
            if (bufferedImage == null) {
                return;
            }
            ff.exporter.setImage(bufferedImage);
            JFrame jFrame = ff.exporter.getFrame();

            if (jFrame != null) {
                jFrame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
                jFrame.setVisible(true);
                jFrame.addWindowListener(new FrameListener(jFrame, ff));
            } else {
                nextPartOfSaving(ff);
            }
        }
    }//GEN-LAST:event_jButton2ActionPerformed
    private Popup popup = new Popup();
    private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton1ActionPerformed
        popup.show(jButton1, 0, jButton1.getHeight());
    }//GEN-LAST:event_jButton1ActionPerformed

    private void jButton3ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton3ActionPerformed
        Object[] elements = jList1.getSelectedValues();
        for (Object e : elements) {
            globalImageListModel.removeElement(e);
            if (globalImageListModel != imageListModel) {
                imageListModel.removeElement(e);
            }
        }
    }//GEN-LAST:event_jButton3ActionPerformed

    protected class Group {

        public Group(String name) {
            this.name = name;
            this.imageListModel = new DefaultListModel();
        }

        public Group() {
            this.imageListModel = new DefaultListModel();
        }

        public Group(String name, DefaultListModel imageListModel) {
            this.name = name;
            this.imageListModel = imageListModel;
        }
        public String name;
        public DefaultListModel imageListModel = null;

        @Override
        public String toString() {
            return name;
        }
    }
    private DefaultComboBoxModel groupsModel = new DefaultComboBoxModel();

    public void addTo(BufferedImage[] images) {
        Group group = new Group();
        renameFrame.setGroup(group);
        renameFrame.setVisible(true);
        for (BufferedImage image : images) {
            group.imageListModel.addElement(new ImageList.Element("", image));
        }
        groupsModel.addElement(group);
    }

    public DefaultListModel createNewGroup() {
        renameFrame.setVisible(true);
        Group group = new Group();
        renameFrame.setGroup(group);
        groupsModel.addElement(group);
        return group.imageListModel;
    }

    public void renameGroup() {
        Group g = (Group) groupsModel.getSelectedItem();
        if (g.imageListModel != globalImageListModel) {
            renameFrame.setVisible(true);
            renameFrame.setGroup(g);
        }
    }

    public void copyTo() {
        copyingFrame.getList().setModel(groupsModel);
        copyingFrame.imageList = this.jList1;
        copyingFrame.setVisible(true);
    }

    public void removeGroup() {
        Group g = (Group) groupsModel.getSelectedItem();
        if (g.imageListModel != globalImageListModel) {
            groupsModel.removeElement(g);
            groupsModel.setSelectedItem(globalGroup);
        }
    }
    private RenameFrame renameFrame = new RenameFrame();
    private CopyingFrame copyingFrame = new CopyingFrame();

    private class JPopup1 extends JPopupMenu {

        private JMenuItem item1 = new JMenuItem("New");
        private JMenuItem item2 = new JMenuItem("Remove");
        private JMenuItem item3 = new JMenuItem("Rename");
        private JMenuItem item4 = new JMenuItem("Copy to");

        public JPopup1() {
            super();
            item1.addActionListener(new ActionListener() {

                public void actionPerformed(ActionEvent e) {
                    createNewGroup();
                }
            });
            item2.addActionListener(new ActionListener() {

                public void actionPerformed(ActionEvent e) {
                    removeGroup();
                }
            });
            item3.addActionListener(new ActionListener() {

                public void actionPerformed(ActionEvent e) {
                    renameGroup();
                }
            });
            item4.addActionListener(new ActionListener() {

                public void actionPerformed(ActionEvent e) {
                    copyTo();
                }
            });
            this.add(item1);
            this.add(item2);
            this.add(item3);
            this.add(item4);
        }
    }
    private JPopup1 jPopup1 = new JPopup1();
    private void jButton4ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton4ActionPerformed
        jPopup1.show(jButton4, 0, jButton4.getHeight());
    }//GEN-LAST:event_jButton4ActionPerformed

    private void jComboBox2ItemStateChanged(java.awt.event.ItemEvent evt) {//GEN-FIRST:event_jComboBox2ItemStateChanged
        Group group = (Group) jComboBox2.getSelectedItem();
        System.out.println(group);
        jList1.setModel(group.imageListModel);
        System.out.println(group.imageListModel);
        imageListModel = group.imageListModel;
    }//GEN-LAST:event_jComboBox2ItemStateChanged

    private void jButton5ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton5ActionPerformed
        ImageList.Element e = (ImageList.Element) jList1.getSelectedValue();
        presentationFrame.setSize(e.bufferedImage.getWidth(), e.bufferedImage.getHeight());
        presentationFrame.setPreferredSize(presentationFrame.getSize());
        presentationFrame.setImage(e.bufferedImage);
        presentationFrame.setVisible(true);
    }//GEN-LAST:event_jButton5ActionPerformed
    private ImageFrame presentationFrame = new ImageFrame("Image");
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton jButton1;
    private javax.swing.JButton jButton2;
    private javax.swing.JButton jButton3;
    private javax.swing.JButton jButton4;
    private javax.swing.JButton jButton5;
    private javax.swing.JComboBox jComboBox2;
    private javax.swing.JList jList1;
    private javax.swing.JList jList2;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JScrollPane jScrollPane2;
    private javax.swing.JToolBar.Separator jSeparator1;
    private javax.swing.JToolBar.Separator jSeparator2;
    private javax.swing.JToolBar.Separator jSeparator3;
    private javax.swing.JToolBar jToolBar1;
    // End of variables declaration//GEN-END:variables
}
