package ogla.core.gui;


import java.util.ArrayList;
import java.util.List;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.DefaultTreeModel;
import ogla.core.application.Application;
import ogla.core.application.ApplicationBundle;
import ogla.core.util.GenericArray;
import ogla.core.view.ViewBundle;

public class ApplicationsFrame extends javax.swing.JFrame {

    private String path = "data/virtualspace/vs_applications";
    private ApplicationsIO applicationsIO = new ApplicationsIO(path);
    private DefaultMutableTreeNode root = new DefaultMutableTreeNode("Applications");
    private DefaultTreeModel treeModel = new DefaultTreeModel(root);
    private String[] info = null;

    public GenericArray getGenericArray() {
        return applicationsIO.getGenericArray();
    }

    public ApplicationsFrame() {
        initComponents();
        jTree1.setModel(treeModel);
        info = applicationsIO.getInfoFromFile();
    }
    private ViewBundle bundleView = null;

    void set(ViewBundle bundleView) {
        if (this.bundleView == null) {
            for (ApplicationLabelContainer fa : applicationLabelContainers) {
                bundleView.isLoadedApplication(fa.application, fa.label);
            }
            applicationLabelContainers.clear();
        } else if (bundleView != this.bundleView) {
            for (ApplicationLabelContainer fa : all) {
                bundleView.isLoadedApplication(fa.application, fa.label);
            }
        }
        this.bundleView = bundleView;
    }

    public final void saveApplications() {
        applicationsIO.save();
    }

    private class BundleApplicationInfo {

        public BundleApplicationInfo(String label, ApplicationBundle bundleApplication) {
            this.label = label;
            this.bundleApplication = bundleApplication;
        }

        public BundleApplicationInfo(ApplicationBundle bundleApplication) {
            this.label = bundleApplication.getLabel();
            this.bundleApplication = bundleApplication;
        }
        public ApplicationBundle bundleApplication;
        public String label;

        @Override
        public String toString() {
            return label;
        }
    }

    private class ApplicationInfo {

        public ApplicationInfo(String symbolicName, String label) {
            this.symbolicName = symbolicName;
            this.label = label;
        }
        public String label;
        public String symbolicName;

        @Override
        public String toString() {
            return label;
        }
    }

    private class ApplicationDateInfo {

        public ApplicationDateInfo(String symbolicName, String label, String date) {
            this.symbolicName = symbolicName;
            this.label = label;
            this.date = date;
        }
        public String symbolicName;
        public String label;
        public String date;
        public byte[] bytes = null;

        @Override
        public String toString() {
            return date;
        }
    }

    public void addApplication(Application application, ApplicationBundle applicationBundle, String label) {
        String date = applicationsIO.addApplication(application, applicationBundle, label);
        register(applicationBundle);
        DefaultMutableTreeNode bundleNode = getNode(applicationBundle, true);
        DefaultMutableTreeNode applicationNode = getApplicationNode(label, bundleNode, applicationBundle);
        DefaultMutableTreeNode node = new DefaultMutableTreeNode(new ApplicationDateInfo(applicationBundle.getSymbolicName(), label, date));
        applicationNode.add(node);
    }
    private List<ApplicationBundle> registeredBundleApplications = new ArrayList<ApplicationBundle>();
    private List<DefaultMutableTreeNode> bundleApplicationNodes = new ArrayList<DefaultMutableTreeNode>();
    private List<DefaultMutableTreeNode> applicationNodes = new ArrayList<DefaultMutableTreeNode>();

    private class ApplicationLabelContainer {

        public ApplicationLabelContainer(Application application, String label) {
            this.application = application;
            this.label = label;
        }
        public Application application;
        public String label;
    }
    private List<ApplicationLabelContainer> applicationLabelContainers = new ArrayList<ApplicationLabelContainer>();
    private List<ApplicationLabelContainer> all = new ArrayList<ApplicationLabelContainer>();

    private void trySendApplicationToBundleView(Application application, String label) {
        ApplicationLabelContainer applicationLabelContainer = new ApplicationLabelContainer(application, label);
        all.add(applicationLabelContainer);
        if (bundleView == null) {
            applicationLabelContainers.add(applicationLabelContainer);
            return;
        }
        bundleView.isLoadedApplication(application, label);
    }

    private void tryInstall(ApplicationBundle applicationBundle) {
        for (ApplicationDateInfo info : applicationDateInfosToLoad) {
            if (info.symbolicName.equals(applicationBundle.getSymbolicName())) {
                Application application = applicationBundle.load(info.bytes);
                trySendApplicationToBundleView(application, info.label);
            }
        }
    }

    private void tryAttachInfoFromFile(ApplicationBundle bundleApplication) {
        if (info == null) {
            return;
        }
        DefaultMutableTreeNode node = getNode(bundleApplication, true);
        for (int fa = 0; fa < info.length; fa += 3) {
            if (info[fa].equals(bundleApplication.getSymbolicName())) {
                DefaultMutableTreeNode applicationNode = getApplicationNode(info[fa + 1], node, bundleApplication);

                DefaultMutableTreeNode childNode = new DefaultMutableTreeNode(
                        new ApplicationDateInfo(bundleApplication.getSymbolicName(), info[fa + 1], info[fa + 2]));
                applicationNode.add(childNode);
            }
        }
        jTree1.revalidate();
    }

    public void register(ApplicationBundle bundleApplication) {
        if (!registeredBundleApplications.contains(bundleApplication)) {
            DefaultMutableTreeNode defaultMutableTreeNode =
                    new DefaultMutableTreeNode(new BundleApplicationInfo(bundleApplication.getLabel(), bundleApplication));
            registeredBundleApplications.add(bundleApplication);
            root.add(defaultMutableTreeNode);
            bundleApplicationNodes.add(defaultMutableTreeNode);
            tryInstall(bundleApplication);
            tryAttachInfoFromFile(bundleApplication);
            jTree1.revalidate();
        }
    }

    private DefaultMutableTreeNode getNode(ApplicationBundle bundleApplication, boolean b) {
        for (DefaultMutableTreeNode node : bundleApplicationNodes) {
            BundleApplicationInfo bundleApplicationInfo = (BundleApplicationInfo) node.getUserObject();
            if (bundleApplicationInfo.bundleApplication == bundleApplication) {
                return node;
            }
        }
        if (b) {
            DefaultMutableTreeNode bundleNode = new DefaultMutableTreeNode(new BundleApplicationInfo(bundleApplication));
            root.add(bundleNode);
            return bundleNode;
        }
        return null;
    }

    private DefaultMutableTreeNode getApplicationNode(String label, DefaultMutableTreeNode bundleNode, ApplicationBundle applicationBundle) {
        for (DefaultMutableTreeNode node : applicationNodes) {
            if (node.getUserObject() instanceof ApplicationDateInfo) {
                ApplicationDateInfo applicationDateInfo = (ApplicationDateInfo) node.getUserObject();
                if (applicationDateInfo.label.equals(label)) {
                    return node;
                }
            }
        }

        DefaultMutableTreeNode appNode = new DefaultMutableTreeNode(new ApplicationInfo(applicationBundle.getSymbolicName(), label));
        bundleNode.add(appNode);
        applicationNodes.add(appNode);
        return appNode;
    }

    public void unregister(ApplicationBundle bundleApplication) {
        if (registeredBundleApplications.contains(bundleApplication)) {
            DefaultMutableTreeNode node = getNode(bundleApplication, false);
            if (node != null) {
                node.removeFromParent();
            }
            jTree1.revalidate();
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

        jToolBar1 = new javax.swing.JToolBar();
        jButton1 = new javax.swing.JButton();
        jButton2 = new javax.swing.JButton();
        jScrollPane1 = new javax.swing.JScrollPane();
        jTree1 = new javax.swing.JTree();

        jToolBar1.setFloatable(false);
        jToolBar1.setRollover(true);

        jButton1.setText("Open");
        jButton1.setFocusable(false);
        jButton1.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jButton1.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        jButton1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton1ActionPerformed(evt);
            }
        });
        jToolBar1.add(jButton1);

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

        javax.swing.tree.DefaultMutableTreeNode treeNode1 = new javax.swing.tree.DefaultMutableTreeNode("root");
        jTree1.setModel(new javax.swing.tree.DefaultTreeModel(treeNode1));
        jTree1.addPropertyChangeListener(new java.beans.PropertyChangeListener() {
            public void propertyChange(java.beans.PropertyChangeEvent evt) {
                jTree1PropertyChange(evt);
            }
        });
        jScrollPane1.setViewportView(jTree1);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jToolBar1, javax.swing.GroupLayout.DEFAULT_SIZE, 400, Short.MAX_VALUE)
            .addComponent(jScrollPane1, javax.swing.GroupLayout.DEFAULT_SIZE, 400, Short.MAX_VALUE)
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(jToolBar1, javax.swing.GroupLayout.PREFERRED_SIZE, 25, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jScrollPane1, javax.swing.GroupLayout.DEFAULT_SIZE, 281, Short.MAX_VALUE))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void jTree1PropertyChange(java.beans.PropertyChangeEvent evt) {//GEN-FIRST:event_jTree1PropertyChange
        // TODO add your handling code here:
    }//GEN-LAST:event_jTree1PropertyChange
    private List<ApplicationDateInfo> applicationDateInfosToLoad = new ArrayList<ApplicationDateInfo>();
    private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton1ActionPerformed
        byte[] bytes = null;
        ApplicationDateInfo applicationDateInfo = null;
        DefaultMutableTreeNode defaultMutableTreeNode = (DefaultMutableTreeNode) jTree1.getSelectionPath().getLastPathComponent();
        if (defaultMutableTreeNode.getUserObject() instanceof ApplicationDateInfo) {
            applicationDateInfo = (ApplicationDateInfo) defaultMutableTreeNode.getUserObject();
            bytes = applicationsIO.load(applicationDateInfo.symbolicName, applicationDateInfo.label, applicationDateInfo.date);
            applicationDateInfo.bytes = bytes;
        }
        boolean wasLoaded = false;
        if (bytes != null) {
            for (ApplicationBundle bundleApplication : registeredBundleApplications) {
                if (bundleApplication.getSymbolicName().equals(applicationDateInfo.symbolicName)) {
                    Application application = bundleApplication.load(bytes);
                    trySendApplicationToBundleView(application, applicationDateInfo.label);
                    wasLoaded = true;
                }
            }
        }
        if (wasLoaded == false && bytes != null) {
            applicationDateInfosToLoad.add(applicationDateInfo);
        }
    }//GEN-LAST:event_jButton1ActionPerformed

    private void jButton2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton2ActionPerformed
        DefaultMutableTreeNode defaultMutableTreeNode = (DefaultMutableTreeNode) jTree1.getSelectionPath().getLastPathComponent();

        if (defaultMutableTreeNode.getUserObject() instanceof ApplicationInfo) {
            ApplicationInfo applicationInfo = (ApplicationInfo) defaultMutableTreeNode.getUserObject();
            applicationsIO.remove(applicationInfo.label, applicationInfo.symbolicName);
            treeModel.removeNodeFromParent(defaultMutableTreeNode);
        }

        if (defaultMutableTreeNode.getUserObject() instanceof ApplicationDateInfo) {
            ApplicationDateInfo applicationDateInfo = (ApplicationDateInfo) defaultMutableTreeNode.getUserObject();
            applicationsIO.remove(applicationDateInfo.label, applicationDateInfo.date, applicationDateInfo.symbolicName);
            treeModel.removeNodeFromParent(defaultMutableTreeNode);
        }
    }//GEN-LAST:event_jButton2ActionPerformed
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton jButton1;
    private javax.swing.JButton jButton2;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JToolBar jToolBar1;
    private javax.swing.JTree jTree1;
    // End of variables declaration//GEN-END:variables
}
