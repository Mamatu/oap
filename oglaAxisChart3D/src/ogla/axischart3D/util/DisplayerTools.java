package ogla.axischart3D.util;

import ogla.data.DataRepository;
import ogla.data.DataBlock;
import ogla.axischart3D.AxisChart3D;
import ogla.axischart3D.plugin.plotmanager.AxisChart3DPlotManager;

public class DisplayerTools {

    public static void create(AxisChart3D axisChart3D, DataRepository[] dataRepositories,
            DisplayerImpl[] displayers, String label, AxisChart3DPlotManager axisChart3DPlotManager) {

        for (int fa = 0; fa < displayers.length; fa++) {
            DisplayerImpl displayer = new DisplayerImpl();

            displayer.setLabel(label);
            for (int fb = 0; fb < dataRepositories.length; fb++) {
                if (dataRepositories[fb] != null) {
                    displayer.dataRepositories.add(dataRepositories[fb]);

                }
            }
            displayers[fa] = displayer;
            axisChart3D.putNewPrimitives(dataRepositories[fa], axisChart3DPlotManager);
        }
    }

    public static void create(AxisChart3D axisChart3D, DataRepository dataRepository,
            DisplayerImpl[] displayers, String label, AxisChart3DPlotManager axisChart3DPlotManager) {
        DataRepository[] array = {dataRepository};
        create(axisChart3D, array, displayers, label, axisChart3DPlotManager);
    }
}
