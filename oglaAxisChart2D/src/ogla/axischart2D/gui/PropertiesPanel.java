package ogla.axischart2D.gui;

import ogla.axischart2D.AxisChart2D;
import ogla.axischart2D.Ticks;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.swing.JTable;
import javax.swing.event.TableModelEvent;
import javax.swing.event.TableModelListener;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.TableModel;

/**
 *
 * @author marcin
 */
public class PropertiesPanel extends javax.swing.JPanel {

    protected AxisChart2D axisChart = null;
    protected DefaultTableModel defaultTableModel;
    private CommandXRange commandXRange = null;
    private CommandYRange commandYRange = null;
    private CommandXTicks commandXTicks = null;
    private CommandYTicks commandYTicks = null;

    /** Creates new form OptionsPanel */
    public PropertiesPanel(AxisChart2D axisChart) {
        initComponents();
        String[] columnsName = {"Properties", "Values"};
        defaultTableModel = new DefaultTableModel(columnsName, 0);
        jTable1.setModel(defaultTableModel);
        this.axisChart = axisChart;
        commandXRange = new CommandXRange();
        commandYRange = new CommandYRange();
        commandXTicks = new CommandXTicks();
        commandYTicks = new CommandYTicks();
        new CommandXLabel();
        new CommandYLabel();
        new CommandXDynamicStep();
        new CommandYDynamicStep();
        new TableListenerImpl();
    }
    private boolean invoke = true;

    public String getColumnName(int column) {
        return jTable1.getColumnName(column);
    }

    public int indexOfValuesColumn() {
        if (jTable1.getColumnName(1).equals("Values")) {
            return 1;
        }
        return 0;
    }

    public void setText(String text, int column, int row) {
        invoke = false;
        jTable1.setValueAt(text, row, column);
        invoke = true;
    }

    public String getText(int column, int row) {
        return (String) jTable1.getValueAt(row, column);
    }

    public void createTicksFromUser() {
        commandXRange.invoke(jTable1, getText(indexOfValuesColumn(), 0), indexOfValuesColumn(), 0);
        commandYRange.invoke(jTable1, getText(indexOfValuesColumn(), 1), indexOfValuesColumn(), 1);
        commandXTicks.invoke(jTable1, getText(indexOfValuesColumn(), 2), indexOfValuesColumn(), 2);
        commandYTicks.invoke(jTable1, getText(indexOfValuesColumn(), 3), indexOfValuesColumn(), 3);
    }

    private class TableListenerImpl
            implements TableModelListener {

        public TableListenerImpl() {
            jTable1.getModel().addTableModelListener(this);
        }

        public void tableChanged(TableModelEvent e) {
            int row = e.getFirstRow();
            int column = e.getColumn();
            TableModel model = (TableModel) e.getSource();
            String data = (String) model.getValueAt(row, column);
            String name = (String) model.getValueAt(row, 0);
            if (invoke && model.getColumnName(column).equals("Values")) {
                commands.get(name).invoke(jTable1, data, column, row);
                axisChart.repaintChartSurface();
            }
        }
    }
    private Map<String, Command> commands = new HashMap<String, Command>();

    private List<Float> parse(String line, Ticks ticks) {

        int i = line.indexOf(":");
        List<Float> out = null;
        Float first = null, end = null, step = null;
        if (i != -1) {
            int i1 = line.indexOf(":", i + 1);

            try {
                if (i1 == -1) {
                    first = new Float(line.substring(0, i));
                    end = new Float(line.substring(i + 1, line.length()));
                    step = 1.f;
                    ticks.setStep(step);
                } else {
                    first = new Float(line.substring(0, i));
                    end = new Float(line.substring(i + 1, i1));
                    step = new Float(line.substring(i1 + 1, line.length()));
                    ticks.setStep(step);
                }
            } catch (Exception exception) {
                return null;
            }
            if (first > end) {
                return null;
            }
        } else {
            return null;
        }
        out = new ArrayList<Float>();
        float current = first;
        int index = 0;
        while (current <= end) {
            out.add(current);
            current += step;
        }
        return out;
    }

    private Float onlyNumber(String line) {
        Float f = null;
        try {
            f = new Float(line);
        } catch (Exception ex) {
            return null;
        }
        return f;
    }

    private List<Float> parseTicks(String line, Ticks ticks) {
        String[] s = line.split(";");
        List<Float> out = new ArrayList<Float>();
        for (int fa = 0; fa < s.length; fa++) {
            Float f = null;
            if ((f = onlyNumber(s[fa])) == null) {
                List<Float> list = parse(s[fa], ticks);
                if (list != null) {
                    for (Float i : list) {
                        out.add(i);
                    }
                }
            } else {
                out.add(f);
            }
        }
        return out;
    }

    private float[] parseScope(String line) {
        int i = line.indexOf(":");
        float[] out;
        if (i != -1) {
            Float low = null, great = null;
            try {
                low = new Float(line.substring(0, i));
                great = new Float(line.substring(i + 1, line.length()));
            } catch (Exception exception) {
                return null;
            }
            if (low > great) {
                return null;
            }
            out = new float[2];
            out[0] = low;
            out[1] = great;
        } else {
            return null;
        }
        return out;
    }

    public interface Command {

        public void invoke(JTable table, String data, int column, int row);
    }

    private class CommandXRange implements Command {

        private float[] x = new float[2];

        public CommandXRange() {
            x[0] = 0;
            x[1] = 10;
            axisChart.getXTicks().setLowestTick(x[0]);
            axisChart.getXTicks().setGreatestTick(x[1]);
            addRow("x range", "0:10", this);
        }

        public void invoke(JTable table, String data, int column, int row) {
            float[] nx = parseScope(data);
            if (nx != null) {
                x[0] = nx[0];
                x[1] = nx[1];
            } else {
                defaultTableModel.setValueAt(new String(x[0] + ":" + x[1]), row, column);
            }
            axisChart.getXTicks().setLowestTick(x[0]);
            axisChart.getXTicks().setGreatestTick(x[1]);
        }
    }

    private class CommandYRange implements Command {

        private float[] y = new float[2];

        public CommandYRange() {
            y[0] = 0;
            y[1] = 10;
            axisChart.getYTicks().setLowestTick(y[0]);
            axisChart.getYTicks().setGreatestTick(y[1]);
            addRow("y range", "0:10", this);
        }

        public void invoke(JTable table, String data, int column, int row) {
            float[] ny = parseScope(data);
            if (ny != null) {
                y[0] = ny[0];
                y[1] = ny[1];
            } else {
                defaultTableModel.setValueAt(new String(y[0] + ":" + y[1]), row, column);
            }
            axisChart.getYTicks().setLowestTick(y[0]);
            axisChart.getYTicks().setGreatestTick(y[1]);
        }
    }

    private class CommandXTicks implements Command {

        private float[] displayedX = new float[10];

        public CommandXTicks() {
            for (int fa = 0; fa < 10; fa++) {
                displayedX[fa] = fa;

            }
            axisChart.getXTicks().setDisplayedTicks(displayedX);
            addRow("x ticks", "0:10:1", this);
        }

        public void invoke(JTable table, String data, int column, int row) {
            List<Float> sx = parseTicks(data, PropertiesPanel.this.axisChart.getXTicks());
            if (sx != null) {
                displayedX = new float[sx.size()];
                for (int fa = 0; fa < sx.size(); fa++) {
                    displayedX[fa] = sx.get(fa);
                }
            }
            axisChart.getXTicks().setDisplayedTicks(displayedX);
        }
    }

    private class CommandYTicks implements Command {

        private float[] displayedY = new float[10];

        public CommandYTicks() {
            for (int fa = 0; fa < 10; fa++) {
                displayedY[fa] = fa;
            }
            axisChart.getYTicks().setDisplayedTicks(displayedY);
            addRow("y ticks", "0:10:1", this);
        }

        public void invoke(JTable table, String data, int column, int row) {
            List<Float> sy = parseTicks(data, PropertiesPanel.this.axisChart.getYTicks());
            if (sy != null) {
                displayedY = new float[sy.size()];
                for (int fa = 0; fa < sy.size(); fa++) {
                    displayedY[fa] = sy.get(fa);
                }
            }
            axisChart.getYTicks().setDisplayedTicks(displayedY);
        }
    }

    private class CommandXLabel implements Command {

        public CommandXLabel() {
            addRow("x length of label", "4", this);
        }

        public void invoke(JTable table, String data, int column, int row) {
            Integer xlength = null;
            try {
                xlength = new Integer(data);
            } catch (NumberFormatException nfe) {
            } finally {
                if (xlength != null) {
                    axisChart.getXTicks().setLengthOfLabels(xlength);
                } else {
                    PropertiesPanel.this.jTable1.setValueAt(axisChart.getXTicks().getLengthOfLabels(), 4, 1);
                }
            }
        }
    }

    private class CommandYLabel implements Command {

        public CommandYLabel() {
            addRow("y length of label", "4", this);
        }

        public void invoke(JTable table, String data, int column, int row) {
            Integer ylength = null;
            try {
                ylength = new Integer(data);
            } catch (NumberFormatException nfe) {
            } finally {
                if (ylength != null) {
                    axisChart.getYTicks().setLengthOfLabels(ylength);
                } else {
                    PropertiesPanel.this.jTable1.setValueAt(axisChart.getYTicks().getLengthOfLabels(), 5, 1);
                }
            }
        }
    }

    private class CommandXDynamicStep implements Command {

        public CommandXDynamicStep() {
            addRow("x dynamic step", "1", this);
        }

        public void invoke(JTable table, String data, int column, int row) {
            Float f = Float.valueOf(data);
            if (f != null) {
                axisChart.getXTicks().setDynamicStep(f);
            }
        }
    }

    private class CommandYDynamicStep implements Command {

        public CommandYDynamicStep() {
            addRow("y dynamic step", "1", this);
        }

        public void invoke(JTable table, String data, int column, int row) {
            Float f = Float.valueOf(data);
            if (f != null) {
                axisChart.getYTicks().setDynamicStep(f);
            }
        }
    }

    public void addRow(String name, String defaultVariable, Command command) {
        String[] o = {name, defaultVariable};
        defaultTableModel.addRow(o);
        jTable1.revalidate();
        jTable1.repaint();
        commands.put(name, command);
    }

    protected javax.swing.JTable getTable() {
        return jTable1;
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
        jTable1 = new javax.swing.JTable();

        setPreferredSize(new java.awt.Dimension(435, 135));

        jTable1.setModel(new javax.swing.table.DefaultTableModel(
            new Object [][] {
                {"x range", "0:10"},
                {"y range", "0:10"},
                {"x ticks", "0:10:1"},
                {"y ticks", "0:10:1"},
                {"x length of labels", "4"},
                {"y length of labels", "4"}
            },
            new String [] {
                "Properties", "Values"
            }
        ) {
            Class[] types = new Class [] {
                java.lang.String.class, java.lang.String.class
            };
            boolean[] canEdit = new boolean [] {
                false, true
            };

            public Class getColumnClass(int columnIndex) {
                return types [columnIndex];
            }

            public boolean isCellEditable(int rowIndex, int columnIndex) {
                return canEdit [columnIndex];
            }
        });
        jScrollPane1.setViewportView(jTable1);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jScrollPane1, javax.swing.GroupLayout.DEFAULT_SIZE, 449, Short.MAX_VALUE)
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jScrollPane1, javax.swing.GroupLayout.PREFERRED_SIZE, 144, javax.swing.GroupLayout.PREFERRED_SIZE)
        );
    }// </editor-fold>//GEN-END:initComponents

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JTable jTable1;
    // End of variables declaration//GEN-END:variables
}
