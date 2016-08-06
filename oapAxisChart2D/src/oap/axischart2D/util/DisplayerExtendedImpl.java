package ogla.axischart2D.util;

import ogla.axischart2D.DisplayerImpl;
import ogla.axischart2D.AxisChart2D;
import ogla.axischart2D.plugin.plotmanager.AxisChart2DPlotManager;
import java.awt.Stroke;
import ogla.core.data.DataRepository;

/**
 *
 * @author marcin
 */
public class DisplayerExtendedImpl extends DisplayerImpl {

    public void setPlotManager(AxisChart2DPlotManager plotManager) {
        this.plotManager = plotManager;
    }

    public void setStroke(Stroke stroke) {
        this.stroke = stroke;
    }

    public DisplayerExtendedImpl(AxisChart2D axisChart, DataRepository dataRepositoriesGroup, String name, AxisIndices axisIndices) {
        super(axisChart, dataRepositoriesGroup, name, axisIndices);
    }

    public DisplayerExtendedImpl(AxisChart2D axisChart, DataRepository[] dataRepositoriesGroups, String name, AxisIndices axisIndices) {
        super(axisChart, dataRepositoriesGroups, name, axisIndices);
    }

    public boolean isCurrent() {
        for (int fa = 0; fa < this.getNumberOfDataRepositories(); fa++) {
            if (getDataRepository(fa).getHistory() != null) {
                return getDataRepository(fa).getHistory().isCurrent();
            }
        }
        return true;
    }

    public void setCurrent() {
        for (int fa = 0; fa < this.getNumberOfDataRepositories(); fa++) {
            drawnDataRepositories[fa] = dataRepositories[fa];
            if (dataRepositories[fa].getHistory() != null) {
                dataRepositories[fa].getHistory().current();
            }
        }
        axisChart.repaintChartSurface();
    }

    public void setTheOldest() {
        for (int fa = 0; fa < this.getNumberOfDataRepositories(); fa++) {
            if (dataRepositories[fa].getHistory() != null) {
                DataRepository dataRepositoriesGroup = dataRepositories[fa].getHistory().getTheOldest();
                if (dataRepositoriesGroup == null) {
                    return;
                }
                drawnDataRepositories[fa] = dataRepositoriesGroup;
            }
        }
        axisChart.repaintChartSurface();
    }

    public void setTheYoungest() {
        for (int fa = 0; fa < this.getNumberOfDataRepositories(); fa++) {
            if (dataRepositories[fa].getHistory() != null) {
                DataRepository dataRepositoriesGroup = dataRepositories[fa].getHistory().getTheYoungest();
                if (dataRepositoriesGroup == null) {
                    return;
                }
                drawnDataRepositories[fa] = dataRepositoriesGroup;
            }
        }
        axisChart.repaintChartSurface();
    }

    public boolean setOlder() {
        for (int fa = 0; fa < this.getNumberOfDataRepositories(); fa++) {
            if (dataRepositories[fa].getHistory() != null) {
                DataRepository dataRepositoriesGroup = dataRepositories[fa].getHistory().older();
                if (dataRepositoriesGroup == null) {
                    return false;
                }
                drawnDataRepositories[fa] = dataRepositoriesGroup;
                axisChart.repaintChartSurface();
                return true;
            }
        }
        axisChart.repaintChartSurface();
        return false;
    }

    public boolean setYounger() {
        for (int fa = 0; fa < this.getNumberOfDataRepositories(); fa++) {
            if (dataRepositories[fa].getHistory() != null) {
                DataRepository dataRepositoriesGroup = dataRepositories[fa].getHistory().younger();
                if (dataRepositoriesGroup == null) {
                    return false;
                }
                drawnDataRepositories[fa] = dataRepositoriesGroup;
                axisChart.repaintChartSurface();
                return true;

            }
        }

        return false;
    }

    public String getDescription() {
        return "";
    }
}
