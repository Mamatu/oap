package ogla.axischart2D.util;

import ogla.axischart2D.DisplayerImpl;
import ogla.axischart.Displayer;
import ogla.axischart.lists.BaseList;
import ogla.axischart2D.AxisChart2D;
import ogla.axischart2D.plugin.plotmanager.AxisChart2DPlotManager;
import ogla.axischart2D.lists.PlotManagerBundleList;
import ogla.core.data.DataRepository;

public final class DisplayerTools {

    public static void create(AxisChart2D axisChart2D, DataRepository[] dataRepositoriesGroups, AxisChart2DPlotManager plotManager,
            DisplayerImpl[] out, String name, AxisIndices axisIndices) {
        DisplayerExtendedImpl displayer =
                new DisplayerExtendedImpl(axisChart2D, dataRepositoriesGroups, name, axisIndices);
        displayer.setPlotManager(plotManager);
        if (axisChart2D.getRepositoryListener() != null) {
            for (DataRepository dataRepositoriesGroup : dataRepositoriesGroups) {
                dataRepositoriesGroup.addRepositoryListener(axisChart2D.getRepositoryListener());
            }
        }
        out[0] = displayer;
    }

    public static void create(AxisChart2D axisChart, DataRepository dataRepositoriesGroup,
            AxisChart2DPlotManager plotManager, DisplayerImpl[] out, String name, AxisIndices axisIndices) {
        DataRepository[] array = new DataRepository[1];
        array[0] = dataRepositoriesGroup;
        create(axisChart, array, plotManager, out, name, axisIndices);
    }

    public static void create(AxisChart2D axisChart2D, DataRepository[] dataRepositoriesGroups,
            String symbolicName, DisplayerImpl[] out, String name, AxisIndices axisIndices) {
        DisplayerExtendedImpl displayer = new DisplayerExtendedImpl(axisChart2D, dataRepositoriesGroups, name, axisIndices);
        if (axisChart2D.getRepositoryListener() != null) {
            for (DataRepository dataRepositoriesGroup : dataRepositoriesGroups) {
                dataRepositoriesGroup.addRepositoryListener(axisChart2D.getRepositoryListener());
            }
        }
        AxisChart2DPlotManager plotManager = null;
        PlotManagerBundleList listPlotManagerBundle = (PlotManagerBundleList) axisChart2D.getListPlotManagerBundle();
        if (listPlotManagerBundle == null) {
            return;
        }
        for (int fa = 0; fa < listPlotManagerBundle.size(); fa++) {
            if (listPlotManagerBundle.get(fa).getSymbolicName().equals(symbolicName)) {
                plotManager = listPlotManagerBundle.get(fa).newPlotManager(axisChart2D);
                break;
            }
        }
        if (plotManager == null) {
            return;
        }
        displayer.setPlotManager(plotManager);
        out[0] = displayer;
    }

    public static void create(AxisChart2D axisChart, DataRepository dataRepositoriesGroup,
            String symbolicName, DisplayerImpl[] out, String name, AxisIndices axisIndices) {
        DataRepository[] array = new DataRepository[1];
        array[0] = dataRepositoriesGroup;
        create(axisChart, array, symbolicName, out, name, axisIndices);
    }

    public static void create(AxisChart2D axisChart2D, DataRepository[] dataRepositoriesGroups, String symbolicName,
            byte[] representation, DisplayerImpl[] out, String name, AxisIndices axisIndices) {
        DisplayerExtendedImpl displayer =
                new DisplayerExtendedImpl(axisChart2D, dataRepositoriesGroups, name, axisIndices);
        if (axisChart2D.getRepositoryListener() != null) {
            for (DataRepository dataRepositoriesGroup : dataRepositoriesGroups) {

                dataRepositoriesGroup.addRepositoryListener(axisChart2D.getRepositoryListener());
            }
        }
        PlotManagerBundleList listPlotManagerBundle = (PlotManagerBundleList) axisChart2D.getListPlotManagerBundle();
        if (listPlotManagerBundle == null) {
            return;
        }
        AxisChart2DPlotManager plotManager = null;
        for (int fa = 0; fa < listPlotManagerBundle.size(); fa++) {
            if (listPlotManagerBundle.get(fa).getSymbolicName().equals(symbolicName)) {
                if (representation == null) {
                    plotManager = listPlotManagerBundle.get(fa).newPlotManager(axisChart2D);
                } else {
                    plotManager = listPlotManagerBundle.get(fa).load(axisChart2D, representation);
                }
                break;
            }
        }
        if (plotManager == null) {
            return;
        }
        displayer.setPlotManager(plotManager);
        out[0] = displayer;
    }

    public static void create(AxisChart2D axisChart2D, DataRepository dataRepositoriesGroup, String symbolicName, byte[] representation,
            DisplayerImpl[] out, String name, AxisIndices axisIndices) {
        DataRepository[] array = new DataRepository[1];
        array[0] = dataRepositoriesGroup;
        create(axisChart2D, array, symbolicName, representation, out, name, axisIndices);

    }
}
