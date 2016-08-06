package ogla.axischart3D;


import java.util.List;
import javax.swing.JPanel;
import ogla.axischart.AxisChartBundle;
import ogla.axischart.lists.BaseList;
import ogla.chart.ChartPanel;
import ogla.core.Help;
import ogla.core.application.Application;
import ogla.core.event.AnalysisEvent;

/**
 *
 * @author marcin
 */
public class AxisChart3DBundle extends AxisChartBundle {

    public Help getHelp() {
        return null;
    }

    public boolean canBeSaved() {
        return false;
    }

    public Application load(byte[] bytes) {
        ApplicationImpl applicationImpl = new ApplicationImpl();
        AxisChart3D axisChart = new AxisChart3D(applicationImpl.desktopPanel,
                baseLists);
        applicationImpl.axisChart = axisChart;
        axisChart.load(bytes);
        return applicationImpl;
    }

    public String getSymbolicName() {
        return "axis_chart_3D_((^^&*#(";
    }

    public AnalysisEvent[] getEvents() {
        return null;
    }

    private class ApplicationImpl implements Application {

        ChartPanel desktopPanel = new ChartPanel();
        AxisChart3D axisChart = null;

        public JPanel getAppPanel() {
            return desktopPanel;
        }

        public byte[] save() {
            return axisChart.save();
        }

        public ApplicationBundle getBundle() {
            return AxisChart3DBundle.this;
        }
    }
    private List<BaseList> baseLists = null;

    public AxisChart3DBundle(List<BaseList> baseLists) {
        this.baseLists = baseLists;
    }

    public Application newApplication() {
        ApplicationImpl applicationImpl = new ApplicationImpl();
        AxisChart3D axisChart = new AxisChart3D(applicationImpl.desktopPanel, baseLists);
        applicationImpl.axisChart = axisChart;
        return applicationImpl;
    }

    public String getLabel() {
        return "Axis Chart 3D";
    }

    public String getDescription() {
        return "";
    }
}
