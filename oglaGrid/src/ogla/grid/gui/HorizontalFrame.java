package ogla.grid.gui;

public class HorizontalFrame extends javax.swing.JFrame {

    /** Creates new form HorizontalFrame */
    public HorizontalFrame() {
        initComponents();
    }

    /** This method is called from within the constructor to
     * initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is
     * always regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        gridPanel1 = new ogla.grid.gui.GridPanel();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        gridPanel1.setBorder(javax.swing.BorderFactory.createTitledBorder("Horizontal"));

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(gridPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, 303, Short.MAX_VALUE)
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(gridPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    public GridPanel getGridPanel() {
        return gridPanel1;
    }
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private ogla.grid.gui.GridPanel gridPanel1;
    // End of variables declaration//GEN-END:variables
}
