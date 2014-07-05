package ogla.core.gui;

import java.awt.AWTException;
import java.awt.Dimension;
import java.awt.SystemTray;
import java.awt.TrayIcon;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import javax.swing.JFileChooser;
import javax.swing.JMenuItem;
import javax.swing.JPopupMenu;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.TreeNode;
import javax.swing.tree.TreePath;
import org.osgi.framework.BundleContext;
import org.osgi.framework.BundleException;

public final class PluginsFrame extends javax.swing.JFrame {

    protected JFileChooser fileChooser = new JFileChooser();
    protected BundleContext bundleContext;
    protected PluginsManager pluginManager = null;
    protected PluginsTree pluginsTree = null;
    final private String pathToPlugins = "data/conf/bundles.conf";
    private IconPopupMenu iconPopupMenu = null;

    private class IconPopupMenu extends JPopupMenu {

        public JMenuItem item1 = new JMenuItem("Close");
        public JMenuItem item2 = new JMenuItem("Hide");

        private class Item1Listener implements ActionListener {

            @Override
            public void actionPerformed(ActionEvent e) {
                System.exit(0);
            }
        }

        private class Item2Listener implements ActionListener {

            @Override
            public void actionPerformed(ActionEvent e) {
                PluginsFrame.this.setVisible(!PluginsFrame.this.isVisible());
            }
        }

        public IconPopupMenu() {
            super();
            this.add(item2);
            this.add(item1);
            this.addMouseListener(new MouseListener() {
                @Override
                public void mouseClicked(MouseEvent e) {
                }

                @Override
                public void mousePressed(MouseEvent e) {
                }

                @Override
                public void mouseReleased(MouseEvent e) {
                    IconPopupMenu.this.setVisible(false);
                }

                @Override
                public void mouseEntered(MouseEvent e) {
                }

                @Override
                public void mouseExited(MouseEvent e) {
                }
            });
            item1.addActionListener(new Item1Listener());
            item2.addActionListener(new Item2Listener());
        }
    }

    @Override
    public void setVisible(boolean b) {
        if (iconPopupMenu != null) {
            if (b) {
                iconPopupMenu.item2.setText("Hide");
            } else {
                iconPopupMenu.item2.setText("Show");
            }
        }
        super.setVisible(b);
    }

    private class SystemTrayNotSupportedException extends Exception {

        public SystemTrayNotSupportedException() {
            super("System Tray is not supported.");
        }
    }

    private void createTrayIcon(BufferedImage bufferedImage) throws IOException, PluginsFrame.SystemTrayNotSupportedException {
        if (SystemTray.isSupported()) {
            iconPopupMenu = new IconPopupMenu();
            TrayIcon trayIcon = new TrayIcon(bufferedImage);
            trayIcon.setPopupMenu(null);
            trayIcon.addMouseListener(new MouseListener() {

                @Override
                public void mouseClicked(MouseEvent e) {
                }

                @Override
                public void mousePressed(MouseEvent e) {
                }

                @Override
                public void mouseReleased(MouseEvent e) {
                    iconPopupMenu.setLocation(e.getX(), e.getY());
                    iconPopupMenu.setInvoker(iconPopupMenu);
                    iconPopupMenu.setVisible(true);
                }

                @Override
                public void mouseEntered(MouseEvent e) {
                }

                @Override
                public void mouseExited(MouseEvent e) {
                }
            });
            try {
                SystemTray.getSystemTray().add(trayIcon);
            } catch (AWTException ex) {
                Logger.getLogger(PluginsFrame.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {
            throw new PluginsFrame.SystemTrayNotSupportedException();
        }
    }

    public PluginsFrame(BundleContext bundleContext, PluginsManager pluginManager, PluginsTree pluginsTree) {
        initComponents();
        this.bundleContext = bundleContext;
        jTree1.setModel(pluginsTree.defaultTreeModel);
        this.pluginManager = pluginManager;
        this.pluginsTree = pluginsTree;
        jTree1.setEditable(true);
        fileChooser.setMultiSelectionEnabled(true);
        this.load();
        try {
            BufferedImage smallImage = ImageIO.read(this.getClass().getResource("ogla_icon_small.png"));
            createTrayIcon(smallImage);
        } catch (IOException ex) {
            System.out.println("Warning: icon is not found.");
        } catch (SystemTrayNotSupportedException ex) {
            System.out.println("Warning: Your system does not support tray.");
        }
    }

    void load() {
        FileInputStream fileInputStream = null;
        File file = new File(pathToPlugins);
        if (!file.exists()) {
            return;
        }
        try {
            fileInputStream = new FileInputStream(file);
            String context = "";
            int c;
            while ((c = fileInputStream.read()) != -1) {
                context += (char) c;
            }
            fileInputStream.close();
            if (context != null && context.length() != 0) {
                String[] lines = context.split("\n");
                for (String line : lines) {
                    Plugin plugin = new Plugin(pluginManager);
                    if (plugin.install(line, bundleContext) == false) {
                        System.err.println("Plugin " + line + " can not be installed.");
                    }
                }
                pluginManager.start();
                checkErrors();
                for (Plugin plugin : pluginManager.installedPlugins) {
                    pluginsTree.add(plugin);
                }
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(PluginsManager.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(PluginsManager.class.getName()).log(Level.SEVERE, null, ex);
        } catch (Exception ex) {
            Logger.getLogger(PluginsManager.class.getName()).log(Level.SEVERE, null, ex);
        } finally {
            try {
                fileInputStream.close();
            } catch (IOException ex) {
                Logger.getLogger(PluginsManager.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }

    void save() {
        FileOutputStream fileOutputStream = null;
        try {
            File file = new File(pathToPlugins);
            file.delete();
            file.createNewFile();
            fileOutputStream = new FileOutputStream(file);
            String endOfLine = "\n";
            if (pluginManager.installedPlugins.size() > 0) {
                for (Plugin plugin : pluginManager.installedPlugins) {
                    fileOutputStream.write(plugin.getPath().getBytes());
                    fileOutputStream.write(endOfLine.getBytes());
                }
            } else {
                fileOutputStream.write(new String("").getBytes());
            }
        } catch (IOException ex) {
            Logger.getLogger(PluginsManager.class.getName()).log(Level.SEVERE, null, ex);
        } finally {
            try {
                if (fileOutputStream != null) {
                    fileOutputStream.close();
                }
            } catch (IOException ex) {
                Logger.getLogger(PluginsManager.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        toolBar = new javax.swing.JToolBar();
        jButton10 = new javax.swing.JButton();
        jButton11 = new javax.swing.JButton();
        jSplitPane6 = new javax.swing.JSplitPane();
        jPanel7 = new javax.swing.JPanel();
        jToolBar3 = new javax.swing.JToolBar();
        jButton2 = new javax.swing.JButton();
        jButton9 = new javax.swing.JButton();
        jButton3 = new javax.swing.JButton();
        jCheckBox1 = new javax.swing.JCheckBox();
        jScrollPane12 = new javax.swing.JScrollPane();
        jTextArea = new javax.swing.JTextArea();
        jScrollPane1 = new javax.swing.JScrollPane();
        jTree1 = new javax.swing.JTree();

        jButton10.setText("Refresh");
        jButton10.setFocusable(false);
        jButton10.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jButton10.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        toolBar.add(jButton10);

        jButton11.setText("Uninstall");
        jButton11.setFocusable(false);
        jButton11.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jButton11.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        toolBar.add(jButton11);

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("Plugins");
        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosed(java.awt.event.WindowEvent evt) {
                formWindowClosed(evt);
            }
            public void windowClosing(java.awt.event.WindowEvent evt) {
                formWindowClosing(evt);
            }
            public void windowOpened(java.awt.event.WindowEvent evt) {
                formWindowOpened(evt);
            }
        });

        jPanel7.setPreferredSize(new java.awt.Dimension(200, 298));

        jToolBar3.setFloatable(false);

        jButton2.setText("Install");
        jButton2.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton2ActionPerformed(evt);
            }
        });
        jToolBar3.add(jButton2);

        jButton9.setText("Uninstall");
        jButton9.setFocusable(false);
        jButton9.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jButton9.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        jButton9.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton9ActionPerformed(evt);
            }
        });
        jToolBar3.add(jButton9);

        jButton3.setText("Active");
        jButton3.setToolTipText("Active installed plugins.");
        jButton3.setFocusable(false);
        jButton3.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jButton3.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        jToolBar3.add(jButton3);

        jCheckBox1.setSelected(true);
        jCheckBox1.setText("Always active");
        jCheckBox1.setFocusable(false);
        jCheckBox1.setHorizontalTextPosition(javax.swing.SwingConstants.RIGHT);
        jCheckBox1.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        jToolBar3.add(jCheckBox1);

        jTextArea.setEditable(false);
        jTextArea.setColumns(20);
        jTextArea.setRows(5);
        jScrollPane12.setViewportView(jTextArea);

        javax.swing.GroupLayout jPanel7Layout = new javax.swing.GroupLayout(jPanel7);
        jPanel7.setLayout(jPanel7Layout);
        jPanel7Layout.setHorizontalGroup(
            jPanel7Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jScrollPane12, javax.swing.GroupLayout.DEFAULT_SIZE, 661, Short.MAX_VALUE)
            .addComponent(jToolBar3, javax.swing.GroupLayout.DEFAULT_SIZE, 661, Short.MAX_VALUE)
        );
        jPanel7Layout.setVerticalGroup(
            jPanel7Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel7Layout.createSequentialGroup()
                .addComponent(jToolBar3, javax.swing.GroupLayout.PREFERRED_SIZE, 25, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jScrollPane12, javax.swing.GroupLayout.DEFAULT_SIZE, 311, Short.MAX_VALUE))
        );

        jSplitPane6.setRightComponent(jPanel7);

        javax.swing.tree.DefaultMutableTreeNode treeNode1 = new javax.swing.tree.DefaultMutableTreeNode("Plugins");
        jTree1.setModel(new javax.swing.tree.DefaultTreeModel(treeNode1));
        jTree1.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                jTree1MouseClicked(evt);
            }
        });
        jScrollPane1.setViewportView(jTree1);

        jSplitPane6.setLeftComponent(jScrollPane1);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jSplitPane6, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, 742, Short.MAX_VALUE)
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(jSplitPane6, javax.swing.GroupLayout.DEFAULT_SIZE, 342, Short.MAX_VALUE)
                .addGap(53, 53, 53))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents
    private final ErrorFrame errorFrame = new ErrorFrame();

    private void checkErrors() {
        String msg = pluginManager.getErrorMsg();
        if (msg.length() > 0) {
            errorFrame.setPluginsManager(pluginManager);
            errorFrame.setText(msg);
            errorFrame.setVisible(true);
        }
    }
    private void jButton2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton2ActionPerformed
        int returnVal = fileChooser.showOpenDialog(this);
        if (returnVal == JFileChooser.APPROVE_OPTION) {
            File[] files = fileChooser.getSelectedFiles();
            if (files.length == 0) {
                files = new File[1];
                files[0] = fileChooser.getSelectedFile();
            }
            for (File file : files) {
                Plugin plugin = new Plugin(pluginManager);
                if (plugin.install(file.getAbsolutePath(), bundleContext)) {
                    pluginsTree.add(plugin);
                }
                pluginManager.start();
                checkErrors();
            }
            jTree1.revalidate();
            jTree1.repaint();
        }
    }//GEN-LAST:event_jButton2ActionPerformed

    private void formWindowClosed(java.awt.event.WindowEvent evt) {//GEN-FIRST:event_formWindowClosed
        this.save();
    }//GEN-LAST:event_formWindowClosed

    private void jButton9ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton9ActionPerformed
        try {
            TreePath treePath = jTree1.getSelectionPath();
            TreeNode treeNode = (TreeNode) treePath.getLastPathComponent();
            DefaultMutableTreeNode defaultMutableTreeNode = (DefaultMutableTreeNode) treeNode;
            ((Plugin) defaultMutableTreeNode.getUserObject()).uninstall();
            pluginsTree.remove((Plugin) defaultMutableTreeNode.getUserObject());
        } catch (BundleException ex) {
            Logger.getLogger(PluginsFrame.class.getName()).log(Level.SEVERE, null, ex);
        }
}//GEN-LAST:event_jButton9ActionPerformed

    private String addStr(String raw, String appendix) {
        if (!raw.equals("")) {
            return appendix + ": " + raw + "\n";
        }
        return "";
    }

    private void jTree1MouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_jTree1MouseClicked
        if (evt.getButton() == MouseEvent.BUTTON1) {
            TreePath path = jTree1.getClosestPathForLocation(evt.getX(), evt.getY());
            jTree1.setSelectionPath(path);
            if (path == null) {
                return;
            }
            Plugin plugin = pluginsTree.get(path);
            if (plugin != null) {
                String displayedInfo = "";
                displayedInfo += addStr(plugin.getName(), "Name");
                displayedInfo += addStr(plugin.getDescription(), "Description");
                displayedInfo += addStr(plugin.getAuthor(), "Author");
                displayedInfo += addStr(plugin.getVersion(), "Version");
                displayedInfo += addStr(plugin.getState(), "State");
                displayedInfo += addStr(plugin.getContact(), "Contact");
                displayedInfo += addStr(plugin.getWebHome(), "Web home");
                displayedInfo += addStr(plugin.getPath(), "Path");
                jTextArea.setText(displayedInfo);
            }
        }
    }//GEN-LAST:event_jTree1MouseClicked

    private void formWindowOpened(java.awt.event.WindowEvent evt) {//GEN-FIRST:event_formWindowOpened
        jTree1.revalidate();
        jTree1.repaint();
    }//GEN-LAST:event_formWindowOpened

    private void formWindowClosing(java.awt.event.WindowEvent evt) {//GEN-FIRST:event_formWindowClosing
        this.save();
    }//GEN-LAST:event_formWindowClosing

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton jButton10;
    private javax.swing.JButton jButton11;
    private javax.swing.JButton jButton2;
    private javax.swing.JButton jButton3;
    private javax.swing.JButton jButton9;
    private javax.swing.JCheckBox jCheckBox1;
    private javax.swing.JPanel jPanel7;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JScrollPane jScrollPane12;
    private javax.swing.JSplitPane jSplitPane6;
    protected javax.swing.JTextArea jTextArea;
    private javax.swing.JToolBar jToolBar3;
    private javax.swing.JTree jTree1;
    private javax.swing.JToolBar toolBar;
    // End of variables declaration//GEN-END:variables
}
