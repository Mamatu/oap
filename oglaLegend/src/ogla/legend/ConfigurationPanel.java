package ogla.legend;

import ogla.axischart.AxisChart;
import ogla.axischart.Displayer;
import ogla.chart.Chart;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import javax.swing.ListModel;
import javax.swing.event.ListDataListener;

/**
 *
 * @author marcin
 */
public class ConfigurationPanel extends javax.swing.JPanel {

    private AxisChart axisChart;

    public ConfigurationPanel(Chart chart, AxisChart axisChartInfo) {
        initComponents();
        jList1.setModel(defaultListModel);
        updateList();
    }
    protected Map<Displayer, String> map = new HashMap<Displayer, String>();
    private ListModel defaultListModel = new ListModel() {

        public int getSize() {
            return map.size();
        }

        public Object getElementAt(int index) {
            int counter = 0;
            String element = null;
            Iterator<Map.Entry<Displayer, String>> iter = map.entrySet().iterator();
            if (iter.hasNext()) {
                do {
                    Map.Entry<Displayer, String> entry = iter.next();
                    element = entry.getValue();
                    counter++;
                } while (iter.hasNext() && counter != index);
            }

            return element;
        }

        public void addListDataListener(ListDataListener l) {
        }

        public void removeListDataListener(ListDataListener l) {
        }
    };

    public void updateList() {
        for (int fa = 0; fa < axisChart.getListDisplayerInfo().size(); fa++) {
            Displayer displayer = axisChart.getListDisplayerInfo().get(fa);
            if (!map.containsKey(displayer)) {
                map.put(displayer, displayer.getDataRepository().getLabel());
            }
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

        jScrollPane1 = new javax.swing.JScrollPane();
        jList1 = new javax.swing.JList();
        jTextField1 = new javax.swing.JTextField();

        addComponentListener(new java.awt.event.ComponentAdapter() {
            public void componentShown(java.awt.event.ComponentEvent evt) {
                formComponentShown(evt);
            }
        });

        jList1.addListSelectionListener(new javax.swing.event.ListSelectionListener() {
            public void valueChanged(javax.swing.event.ListSelectionEvent evt) {
                jList1ValueChanged(evt);
            }
        });
        jScrollPane1.setViewportView(jList1);

        jTextField1.addPropertyChangeListener(new java.beans.PropertyChangeListener() {
            public void propertyChange(java.beans.PropertyChangeEvent evt) {
                jTextField1PropertyChange(evt);
            }
        });
        jTextField1.addKeyListener(new java.awt.event.KeyAdapter() {
            public void keyReleased(java.awt.event.KeyEvent evt) {
                jTextField1KeyReleased(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(jScrollPane1, javax.swing.GroupLayout.PREFERRED_SIZE, 108, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(24, 24, 24)
                .addComponent(jTextField1, javax.swing.GroupLayout.DEFAULT_SIZE, 201, Short.MAX_VALUE)
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGap(12, 12, 12)
                .addComponent(jTextField1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(260, Short.MAX_VALUE))
            .addComponent(jScrollPane1, javax.swing.GroupLayout.DEFAULT_SIZE, 300, Short.MAX_VALUE)
        );
    }// </editor-fold>//GEN-END:initComponents

    private Displayer get(int index) {
        int counter = 0;
        Displayer out = null;
        Iterator<Displayer> iter = map.keySet().iterator();
        if (iter.hasNext()) {
            do {
                out = iter.next();
                counter++;
            } while (iter.hasNext() && counter != index);
        }
        return out;
    }
    private void jList1ValueChanged(javax.swing.event.ListSelectionEvent evt) {//GEN-FIRST:event_jList1ValueChanged
        int index = jList1.getSelectedIndex();
        if (index == -1) {
            return;
        }
        String text = (String) defaultListModel.getElementAt(index);
        jTextField1.setText(text);
    }//GEN-LAST:event_jList1ValueChanged

    private void jTextField1PropertyChange(java.beans.PropertyChangeEvent evt) {//GEN-FIRST:event_jTextField1PropertyChange
        int index = jList1.getSelectedIndex();
        if (index == -1) {
            return;
        }
        String text = (String) evt.getNewValue();
        map.put(get(index), text);

    }//GEN-LAST:event_jTextField1PropertyChange

    private void formComponentShown(java.awt.event.ComponentEvent evt) {//GEN-FIRST:event_formComponentShown
        updateList();
    }//GEN-LAST:event_formComponentShown

    private void jTextField1KeyReleased(java.awt.event.KeyEvent evt) {//GEN-FIRST:event_jTextField1KeyReleased
        if (evt.getKeyCode() == java.awt.event.KeyEvent.VK_ENTER) {
            int index = jList1.getSelectedIndex();
            if (index == -1) {
                return;
            }
            String text = (String) jTextField1.getText();
            map.put(get(index), text);
            jList1.revalidate();
            jList1.repaint();
        }
    }//GEN-LAST:event_jTextField1KeyReleased
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JList jList1;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JTextField jTextField1;
    // End of variables declaration//GEN-END:variables
}
